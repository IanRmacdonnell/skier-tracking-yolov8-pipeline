import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

LOCAL_RUNTIME_DIR = Path(".runtime")
ULTRALYTICS_CONFIG_DIR = LOCAL_RUNTIME_DIR / "ultralytics"
MATPLOTLIB_CONFIG_DIR = LOCAL_RUNTIME_DIR / "matplotlib"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR.resolve()))

import cv2
import numpy as np
from ultralytics import YOLO

from tracker_math import box_center, box_from_center_size, box_iou, box_size, clamp_box, expand_box, smooth_box, xywh_to_xyxy, xyxy_to_xywh
from tracking_io import open_csv_logger, open_video_writer, write_summary


PERSON_CLASS = "person"
SKIER_CLASS = "skier"
SKIER_EQUIPMENT_CLASSES = {"skis", "snowboard"}
YOLO_CLASSES = [0, 30, 31]


@dataclass
class Detection:
    xyxy: np.ndarray
    cls_name: str
    confidence: float
    track_id: Optional[int]

    @property
    def center(self) -> np.ndarray:
        return box_center(self.xyxy)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


@dataclass
class Candidate:
    detection: Detection
    score: float
    equipment_seen: bool
    appearance_score: float
    distance_ratio: float
    area_ratio: float
    iou: float
    flow_iou: float
    accepted: bool
    reject_reason: str = ""
    subject_bonus: float = 0.0


@dataclass
class PrimarySkier:
    token_id: int
    box: np.ndarray
    confidence: float
    source_track_id: Optional[int] = None
    equipment_seen: bool = False
    missed_frames: int = 0
    detection_updates: int = 0
    tracker_updates: int = 0
    tracker_gap: int = 0
    rejected_jumps: int = 0
    age: int = 1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    size_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    appearance_hist: Optional[np.ndarray] = None
    appearance_gallery: List[np.ndarray] = field(default_factory=list)
    history: List[Tuple[int, int]] = field(default_factory=list)
    last_source: str = "init"
    state: str = "LOCKED"
    uncertain_frames: int = 0
    reacquire_frames: int = 0
    flow_points: Optional[np.ndarray] = None
    prev_gray: Optional[np.ndarray] = None
    pending_reacquire_box: Optional[np.ndarray] = None
    pending_reacquire_hits: int = 0
    rejected_reacquire: int = 0

    def update_from_detection(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        confidence: float,
        source_track_id: Optional[int],
        equipment_seen: bool,
        smoothing: float,
        appearance_alpha: float,
    ) -> None:
        old_center = box_center(self.box)
        new_center = box_center(box)
        old_size = box_size(self.box)
        new_size = box_size(box)
        self.velocity = (0.65 * self.velocity) + (0.35 * (new_center - old_center))
        self.size_velocity = (0.70 * self.size_velocity) + (0.30 * (new_size - old_size))
        self.box = smooth_box(self.box, box, smoothing)
        self.confidence = confidence
        self.source_track_id = source_track_id or self.source_track_id
        self.equipment_seen = self.equipment_seen or equipment_seen
        self.missed_frames = 0
        self.detection_updates += 1
        self.tracker_gap = 0
        self.age += 1
        self.last_source = "yolo"
        self.state = "LOCKED"
        self.uncertain_frames = 0
        self.reacquire_frames = 0
        self.pending_reacquire_box = None
        self.pending_reacquire_hits = 0
        self._refresh_history()
        self.update_appearance(frame, self.box, appearance_alpha)
        self.reset_optical_flow(frame)

    def update_from_tracker(self, box: np.ndarray, smoothing: float, source: str = "opencv") -> None:
        old_center = box_center(self.box)
        new_center = box_center(box)
        old_size = box_size(self.box)
        new_size = box_size(box)
        self.velocity = (0.70 * self.velocity) + (0.30 * (new_center - old_center))
        self.size_velocity = (0.75 * self.size_velocity) + (0.25 * (new_size - old_size))
        self.box = smooth_box(self.box, box, smoothing)
        self.missed_frames += 1
        self.tracker_updates += 1
        self.tracker_gap += 1
        self.age += 1
        self.last_source = source
        self._update_uncertainty()
        self._refresh_history()

    def update_from_manual_box(self, frame: np.ndarray, box: np.ndarray, source: str = "manual", smoothing: float = 0.0) -> None:
        old_center = box_center(self.box)
        new_center = box_center(box)
        old_size = box_size(self.box)
        new_size = box_size(box)
        self.velocity = new_center - old_center
        self.size_velocity = new_size - old_size
        self.box = smooth_box(self.box, box.astype(float), smoothing) if smoothing > 0 else box.astype(float)
        self.confidence = 1.0
        self.missed_frames = 0
        self.tracker_gap = 0
        self.age += 1
        self.last_source = source
        self.state = "LOCKED"
        self.uncertain_frames = 0
        self.reacquire_frames = 0
        self.pending_reacquire_box = None
        self.pending_reacquire_hits = 0
        self._refresh_history()
        self.update_appearance(frame, self.box, alpha=1.0)
        self.reset_optical_flow(frame)

    def predict_box(self) -> np.ndarray:
        cx, cy = box_center(self.box) + self.velocity
        width, height = np.maximum(1.0, box_size(self.box) + self.size_velocity)
        return box_from_center_size(cx, cy, width, height)

    def mark_missed(self) -> None:
        self.missed_frames += 1
        self.tracker_gap += 1
        self.age += 1
        self.last_source = "prediction"
        self._update_uncertainty()
        self._refresh_history()

    def update_appearance(self, frame: np.ndarray, box: np.ndarray, alpha: float) -> None:
        hist = compute_appearance_hist(frame, box)
        if hist is None:
            return
        if self.appearance_hist is None:
            self.appearance_hist = hist
        else:
            self.appearance_hist = ((1.0 - alpha) * self.appearance_hist) + (alpha * hist)
            cv2.normalize(self.appearance_hist, self.appearance_hist, 0, 1, cv2.NORM_MINMAX)
        self.appearance_gallery.append(hist)
        self.appearance_gallery = self.appearance_gallery[-12:]

    def reset_optical_flow(self, frame: np.ndarray) -> None:
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(self.prev_gray.shape, dtype=np.uint8)
        x1, y1, x2, y2 = clamp_box(self.box, frame.shape).astype(int)
        mask[y1:y2, x1:x2] = 255
        self.flow_points = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=40,
            qualityLevel=0.01,
            minDistance=3,
            blockSize=5,
            mask=mask,
        )

    def estimate_optical_flow_box(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.prev_gray is None or self.flow_points is None or len(self.flow_points) < 4:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.flow_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        self.prev_gray = gray
        if next_points is None or status is None:
            self.flow_points = None
            return None
        good_old = self.flow_points[status.reshape(-1) == 1]
        good_new = next_points[status.reshape(-1) == 1]
        self.flow_points = good_new.reshape(-1, 1, 2) if len(good_new) >= 4 else None
        if len(good_new) < 4:
            return None
        delta = np.median((good_new - good_old).reshape(-1, 2), axis=0)
        if np.linalg.norm(delta) > max(90.0, box_size(self.box)[1] * 3.0):
            return None
        return clamp_box(self.box + np.array([delta[0], delta[1], delta[0], delta[1]], dtype=float), frame.shape)

    def _refresh_history(self) -> None:
        center = box_center(self.box)
        self.history.append((int(center[0]), int(center[1])))
        self.history = self.history[-96:]

    def _update_uncertainty(self) -> None:
        if self.missed_frames == 0:
            self.state = "LOCKED"
            self.uncertain_frames = 0
            self.reacquire_frames = 0
        elif self.missed_frames <= 6:
            self.state = "UNCERTAIN"
            self.uncertain_frames += 1
        elif self.missed_frames <= 30:
            self.state = "REACQUIRE"
            self.reacquire_frames += 1
        else:
            self.state = "LOST"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track one selected skier through a video.")
    parser.add_argument("--video", default="videos/ski.mp4", help="Input video path.")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model path or model name.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--conf", type=float, default=0.18, help="Detection confidence threshold.")
    parser.add_argument("--skier-conf", type=float, default=0.05, help="Detection confidence threshold used automatically for custom one-class skier models.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--skier-imgsz", type=int, default=640, help="Inference image size used automatically for custom one-class skier models.")
    parser.add_argument("--yolo-every", type=int, default=1, help="Run YOLO every N frames. Default 1 means frame-by-frame detection.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for quick validation runs.")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Saved/preview playback speed. Example: 1.5 or 2.0.")
    parser.add_argument("--start-at", default="", help="Start at a frame/time, for example 450, 15s, 1:25, or 00:01:25.")
    parser.add_argument("--end-at", default="", help="Stop at a frame/time, for example 1200, 45s, 2:10, or 00:02:10.")
    parser.add_argument("--skip-ranges", default="", help="Comma-separated frame/time ranges to skip, for example 1:00-1:15,900-1050.")
    parser.add_argument("--output", default="output/skier_tracking.mp4", help="Annotated output video path.")
    parser.add_argument("--no-display", action="store_true", help="Run without opening a preview window.")
    parser.add_argument("--save", action="store_true", help="Save an annotated output video.")
    parser.add_argument("--select-first-frame", action="store_true", help="Draw the skier box on the first frame.")
    parser.add_argument("--manual-every", type=int, default=0, help="Manually correct the skier box every N frames. Use 1 for true frame-by-frame manual tracking.")
    parser.add_argument("--init-box", default="", help="Initial skier box as x1,y1,x2,y2 for headless runs.")
    parser.add_argument("--keyframes", default="", help="Optional keyframe CSV to guide/re-anchor tracking through turns.")
    parser.add_argument("--keyframe-blend", type=float, default=0.15, help="Smoothing for keyframe/interpolated boxes. 0 uses labels directly.")
    parser.add_argument("--log-csv", default="", help="Optional CSV path for per-frame tracking diagnostics.")
    parser.add_argument("--summary", default="output/tracking_summary.json", help="JSON summary path used when saving/logging.")
    parser.add_argument("--max-missed", type=int, default=90, help="Frames to keep predicting without YOLO reacquisition.")
    parser.add_argument("--max-tracker-gap", type=int, default=3, help="Max consecutive OpenCV-only frames before requiring YOLO.")
    parser.add_argument("--smooth", type=float, default=0.68, help="EMA smoothing for accepted YOLO boxes.")
    parser.add_argument("--tracker-smooth", type=float, default=0.82, help="EMA smoothing for OpenCV tracker boxes.")
    parser.add_argument("--appearance-weight", type=float, default=0.45, help="How strongly crop appearance affects reacquisition.")
    parser.add_argument("--appearance-threshold", type=float, default=0.45, help="Minimum appearance match for cautious reacquisition.")
    parser.add_argument("--max-jump", type=float, default=1.2, help="Max center jump as multiples of current box height.")
    parser.add_argument("--max-area-change", type=float, default=2.8, help="Max area ratio change for accepted detections.")
    parser.add_argument("--min-iou", type=float, default=0.02, help="Minimum IoU with predicted/search box for reacquisition.")
    parser.add_argument("--turn-search-scale", type=float, default=4.0, help="Search expansion while uncertain or reacquiring through turns.")
    parser.add_argument("--flow-weight", type=float, default=0.35, help="How strongly optical-flow prediction affects candidate scoring.")
    parser.add_argument("--reacquire-hits", type=int, default=2, help="Repeated candidate hits required before accepting a weak reacquisition.")
    parser.add_argument("--global-reacquire-appearance", type=float, default=0.85, help="Appearance match needed to accept a custom skier detection after a large jump.")
    parser.add_argument("--global-reacquire-conf", type=float, default=0.10, help="Confidence needed to accept a custom skier detection after a large jump.")
    parser.add_argument("--global-reacquire-area", type=float, default=10.0, help="Largest area ratio allowed for custom skier global reacquisition.")
    parser.add_argument("--use-opencv-tracker", action=argparse.BooleanOptionalAction, default=True, help="Bridge YOLO gaps with an OpenCV tracker.")
    parser.add_argument("--opencv-tracker", default="MIL", help="Preferred OpenCV tracker: CSRT, KCF, MIL, or MOSSE.")
    parser.add_argument("--multi-person", action="store_true", help="Debug mode: draw all person detections as candidates.")
    return parser.parse_args()


def parse_frame_or_time(value: str, fps: float) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    if value.lower().endswith("s"):
        seconds = float(value[:-1])
        return max(1, int(round(seconds * fps)) + 1)
    if ":" in value:
        parts = [float(part) for part in value.split(":")]
        if len(parts) == 2:
            seconds = (parts[0] * 60.0) + parts[1]
        elif len(parts) == 3:
            seconds = (parts[0] * 3600.0) + (parts[1] * 60.0) + parts[2]
        else:
            raise ValueError(f"Invalid time value: {value}")
        return max(1, int(round(seconds * fps)) + 1)
    return max(1, int(value))


def parse_skip_ranges(value: str, fps: float) -> List[Tuple[int, int]]:
    ranges = []
    if not value.strip():
        return ranges
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Skip range must be formatted as start-end: {item}")
        start_value, end_value = item.split("-", 1)
        start = parse_frame_or_time(start_value, fps)
        end = parse_frame_or_time(end_value, fps)
        if start is None or end is None:
            continue
        start, end = sorted((start, end))
        ranges.append((start, end))
    return sorted(ranges)


def skipped_range_end(frame_number: int, ranges: List[Tuple[int, int]]) -> Optional[int]:
    for start, end in ranges:
        if start <= frame_number <= end:
            return end
    return None


def parse_init_box(value: str, frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--init-box must be formatted as x1,y1,x2,y2")
    return clamp_box(np.array([float(part) for part in parts], dtype=float), frame_shape)


def parse_detections(result, names: Dict[int, str]) -> List[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)

    detections = []
    for box, cls_id, confidence, track_id in zip(xyxy, classes, confidences, ids):
        cls_name = names.get(int(cls_id), str(cls_id))
        if is_subject_class(cls_name) or cls_name in SKIER_EQUIPMENT_CLASSES:
            detections.append(
                Detection(
                    xyxy=box.astype(float),
                    cls_name=cls_name,
                    confidence=float(confidence),
                    track_id=None if track_id is None else int(track_id),
                )
            )
    return detections


def is_subject_class(cls_name: str) -> bool:
    return cls_name.lower() in {PERSON_CLASS, SKIER_CLASS}


def equipment_near_person(person: Detection, equipment: Iterable[Detection], frame_shape: Tuple[int, int, int]) -> bool:
    search_box = expand_box(person.xyxy, frame_shape, scale_x=1.8, scale_y=2.15)
    person_center = person.center
    person_height = max(1.0, person.xyxy[3] - person.xyxy[1])

    for item in equipment:
        if box_iou(search_box, item.xyxy) > 0.01:
            return True
        distance = np.linalg.norm(item.center - person_center)
        if distance < person_height * 1.35 and item.center[1] >= person.xyxy[1]:
            return True
    return False


def compute_appearance_hist(frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
    box = clamp_box(box, frame.shape).astype(int)
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def appearance_gallery_similarity(skier: PrimarySkier, frame: np.ndarray, box: np.ndarray) -> float:
    hist = compute_appearance_hist(frame, box)
    if hist is None:
        return 0.0
    scores = []
    if skier.appearance_hist is not None:
        scores.append(cv2.compareHist(skier.appearance_hist, hist, cv2.HISTCMP_CORREL))
    for reference in skier.appearance_gallery:
        scores.append(cv2.compareHist(reference, hist, cv2.HISTCMP_CORREL))
    if not scores:
        return 0.5
    normalized = [float(np.clip((score + 1.0) / 2.0, 0.0, 1.0)) for score in scores]
    return max(normalized)


def automatic_initial_box(detections: List[Detection], frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    people = [d for d in detections if is_subject_class(d.cls_name)]
    equipment = [d for d in detections if d.cls_name in SKIER_EQUIPMENT_CLASSES]
    if not people:
        return None
    height, width = frame_shape[:2]
    scored = []
    for person in people:
        has_equipment = equipment_near_person(person, equipment, frame_shape)
        cx, cy = person.center
        center_bias = 1.0 - min(1.0, abs(cx - (width / 2.0)) / (width / 2.0))
        lower_bias = min(1.0, cy / max(1.0, height))
        equipment_bonus = 0.55 if has_equipment else 0.0
        score = person.confidence + equipment_bonus + (0.25 * center_bias) + (0.15 * lower_bias)
        scored.append((score, person))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1].xyxy.copy()


def select_initial_box(frame: np.ndarray, args: argparse.Namespace, detections: List[Detection]) -> np.ndarray:
    init_box = parse_init_box(args.init_box, frame.shape)
    if init_box is not None:
        return init_box

    should_select = args.select_first_frame or not args.no_display
    if should_select and not args.no_display:
        preview = frame.copy()
        for detection in detections:
            if is_subject_class(detection.cls_name):
                x1, y1, x2, y2 = detection.xyxy.astype(int)
                cv2.rectangle(preview, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.putText(
            preview,
            "Draw the skier, press ENTER/SPACE. Press C to auto-pick.",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )
        roi = cv2.selectROI("Select Skier", preview, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Skier")
        if roi[2] > 0 and roi[3] > 0:
            return clamp_box(xywh_to_xyxy(roi), frame.shape)

    auto_box = automatic_initial_box(detections, frame.shape)
    if auto_box is not None:
        print("No manual initial box supplied; using the best first-frame person candidate.")
        return auto_box

    raise RuntimeError("Could not initialize skier. Use --init-box x1,y1,x2,y2 or enable --select-first-frame.")


def manually_correct_box(frame: np.ndarray, skier: PrimarySkier, frame_number: int) -> Optional[np.ndarray]:
    preview = frame.copy()
    x1, y1, x2, y2 = skier.box.astype(int)
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 180, 255), 2)
    cv2.putText(
        preview,
        f"Frame {frame_number}: draw skier box. ENTER/SPACE accepts, C keeps current.",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 255),
        2,
    )
    roi = cv2.selectROI("Correct Skier", preview, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Correct Skier")
    if roi[2] <= 0 or roi[3] <= 0:
        return None
    return clamp_box(xywh_to_xyxy(roi), frame.shape)


def load_keyframe_boxes(path: str) -> List[Tuple[int, np.ndarray]]:
    if not path:
        return []
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    int(row["frame"]),
                    np.array(
                        [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                        dtype=float,
                    ),
                )
            )
    rows.sort(key=lambda item: item[0])
    return rows


def interpolated_keyframe_box(frame_number: int, keyframes: List[Tuple[int, np.ndarray]], frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if not keyframes or frame_number < keyframes[0][0] or frame_number > keyframes[-1][0]:
        return None
    exact = {number: box for number, box in keyframes}
    if frame_number in exact:
        return clamp_box(exact[frame_number], frame_shape)

    previous = None
    following = None
    for item in keyframes:
        if item[0] < frame_number:
            previous = item
        elif item[0] > frame_number:
            following = item
            break
    if previous is None or following is None:
        return None
    prev_frame, prev_box = previous
    next_frame, next_box = following
    span = max(1, next_frame - prev_frame)
    t = (frame_number - prev_frame) / span
    return clamp_box(prev_box + ((next_box - prev_box) * t), frame_shape)


def create_opencv_tracker(name: str):
    preferred = name.upper()
    candidates = [preferred, "CSRT", "KCF", "MIL", "MOSSE"]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        creator_names = [f"Tracker{candidate}_create"]
        if hasattr(cv2, "legacy"):
            creator_names.append(f"legacy.Tracker{candidate}_create")
        for creator_name in creator_names:
            try:
                if creator_name.startswith("legacy."):
                    creator = getattr(cv2.legacy, creator_name.split(".", 1)[1])
                else:
                    creator = getattr(cv2, creator_name)
                return creator(), candidate
            except AttributeError:
                continue
            except cv2.error:
                continue
    return None, "none"


def reset_opencv_tracker(frame: np.ndarray, box: np.ndarray, args: argparse.Namespace):
    if not args.use_opencv_tracker:
        return None, "disabled"
    tracker, name = create_opencv_tracker(args.opencv_tracker)
    if tracker is None:
        return None, "unavailable"
    tracker.init(frame, xyxy_to_xywh(box))
    return tracker, name


def evaluate_candidates(
    frame: np.ndarray,
    skier: PrimarySkier,
    detections: List[Detection],
    args: argparse.Namespace,
    flow_box: Optional[np.ndarray] = None,
) -> List[Candidate]:
    people = [d for d in detections if is_subject_class(d.cls_name)]
    equipment = [d for d in detections if d.cls_name in SKIER_EQUIPMENT_CLASSES]
    predicted = clamp_box(skier.predict_box(), frame.shape)
    if flow_box is not None:
        predicted = smooth_box(predicted, flow_box, 0.45)
    search_scale = args.turn_search_scale if skier.state in {"UNCERTAIN", "REACQUIRE", "LOST"} else 2.6
    search_box = expand_box(predicted, frame.shape, scale_x=search_scale, scale_y=search_scale)
    current_area = max(1.0, (skier.box[2] - skier.box[0]) * (skier.box[3] - skier.box[1]))
    current_height = max(1.0, skier.box[3] - skier.box[1])
    predicted_center = box_center(predicted)
    candidates = []

    for person in people:
        has_equipment = equipment_near_person(person, equipment, frame.shape)
        iou = max(box_iou(person.xyxy, predicted), box_iou(person.xyxy, search_box))
        distance_ratio = float(np.linalg.norm(person.center - predicted_center) / current_height)
        area_ratio = float(max(person.area, current_area) / max(1.0, min(person.area, current_area)))
        appearance = appearance_gallery_similarity(skier, frame, person.xyxy)
        same_yolo_track = person.track_id is not None and person.track_id == skier.source_track_id
        custom_skier_detection = person.cls_name.lower() == SKIER_CLASS
        subject_bonus = 0.35 if custom_skier_detection else 0.0
        flow_iou = box_iou(person.xyxy, flow_box) if flow_box is not None else 0.0
        global_reacquire = (
            custom_skier_detection
            and skier.state in {"UNCERTAIN", "REACQUIRE", "LOST"}
            and appearance >= args.global_reacquire_appearance
            and person.confidence >= args.global_reacquire_conf
            and area_ratio <= args.global_reacquire_area
        )

        accepted = True
        reject_reason = ""
        state_jump_multiplier = 2.4 if skier.state in {"REACQUIRE", "LOST"} else 1.45 if skier.state == "UNCERTAIN" else 1.0
        max_jump = args.max_jump * state_jump_multiplier
        forgiving_reacquire = skier.tracker_gap >= args.max_tracker_gap and appearance >= 0.68
        if area_ratio > args.max_area_change * 1.8 and not global_reacquire:
            accepted = False
            reject_reason = "hard-scale"
        elif distance_ratio > max_jump and not same_yolo_track and not forgiving_reacquire and not global_reacquire and flow_iou < 0.10:
            accepted = False
            reject_reason = "jump"
        elif area_ratio > args.max_area_change and not (same_yolo_track and flow_iou >= 0.10) and not global_reacquire:
            accepted = False
            reject_reason = "scale"
        elif iou < args.min_iou and flow_iou < args.min_iou and distance_ratio > 1.0 and not same_yolo_track and not forgiving_reacquire and not global_reacquire:
            accepted = False
            reject_reason = "outside-gate"
        elif appearance < args.appearance_threshold and distance_ratio > 0.9 and not same_yolo_track and not custom_skier_detection:
            accepted = False
            reject_reason = "appearance"

        score = person.confidence
        score += subject_bonus
        score += 0.40 if global_reacquire else 0.0
        score += args.appearance_weight * appearance
        score += 0.35 * max(0.0, 1.0 - min(distance_ratio / max(max_jump, 0.1), 1.0))
        score += 0.25 if has_equipment else 0.0
        score += 0.30 if same_yolo_track else 0.0
        score += 0.20 * min(iou, 1.0)
        score += args.flow_weight * min(flow_iou, 1.0)

        candidates.append(
            Candidate(
                detection=person,
                score=score,
                equipment_seen=has_equipment,
                appearance_score=appearance,
                distance_ratio=distance_ratio,
                area_ratio=area_ratio,
                iou=iou,
                flow_iou=flow_iou,
                accepted=accepted,
                reject_reason=reject_reason,
                subject_bonus=subject_bonus,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def choose_candidate(candidates: List[Candidate]) -> Optional[Candidate]:
    for candidate in candidates:
        if candidate.accepted:
            return candidate
    return None


def confirm_reacquisition(skier: PrimarySkier, candidate: Candidate, args: argparse.Namespace) -> bool:
    if skier.state in {"LOCKED", "UNCERTAIN"}:
        skier.pending_reacquire_box = None
        skier.pending_reacquire_hits = 0
        return True

    strong_custom_detection = (
        candidate.detection.cls_name.lower() == SKIER_CLASS
        and candidate.detection.confidence >= 0.35
        and candidate.appearance_score >= args.appearance_threshold
    )
    strong_motion_agreement = candidate.iou >= 0.25 or candidate.flow_iou >= 0.25
    if strong_custom_detection and strong_motion_agreement:
        skier.pending_reacquire_box = None
        skier.pending_reacquire_hits = 0
        return True

    if skier.pending_reacquire_box is None or box_iou(skier.pending_reacquire_box, candidate.detection.xyxy) < 0.15:
        skier.pending_reacquire_box = candidate.detection.xyxy.copy()
        skier.pending_reacquire_hits = 1
        skier.rejected_reacquire += 1
        return False

    skier.pending_reacquire_hits += 1
    skier.pending_reacquire_box = smooth_box(skier.pending_reacquire_box, candidate.detection.xyxy, 0.45)
    if skier.pending_reacquire_hits >= max(1, args.reacquire_hits):
        skier.pending_reacquire_box = None
        skier.pending_reacquire_hits = 0
        return True

    skier.rejected_reacquire += 1
    return False


def draw_skier_overlay(
    frame: np.ndarray,
    skier: Optional[PrimarySkier],
    candidates: List[Candidate],
    args: argparse.Namespace,
    frame_number: int,
    tracker_name: str,
) -> np.ndarray:
    overlay = frame.copy()
    if args.multi_person:
        for candidate in candidates:
            x1, y1, x2, y2 = candidate.detection.xyxy.astype(int)
            color = (80, 180, 80) if candidate.accepted else (80, 80, 220)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            label = f"{candidate.score:.2f} app:{candidate.appearance_score:.2f}"
            if not candidate.accepted:
                label += f" reject:{candidate.reject_reason}"
            cv2.putText(overlay, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if skier is not None:
        x1, y1, x2, y2 = skier.box.astype(int)
        color = (0, 255, 255) if skier.missed_frames == 0 else (0, 170, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        label = f"skier:{skier.token_id} {skier.state} source:{skier.last_source} conf:{skier.confidence:.2f} missed:{skier.missed_frames}"
        if skier.equipment_seen:
            label += " equipment"
        cv2.putText(overlay, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
        if len(skier.history) > 1:
            points = np.array(skier.history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [points], False, color, 2)

    cv2.putText(
        overlay,
        f"frame:{frame_number} tracker:{tracker_name} candidates:{len(candidates)}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )
    return overlay


def write_frame_log(writer, frame_number: int, skier: PrimarySkier, candidates: List[Candidate], accepted: Optional[Candidate]) -> None:
    if writer is None:
        return
    x1, y1, x2, y2 = skier.box
    writer.writerow(
        {
            "frame": frame_number,
            "state": skier.state,
            "source": skier.last_source,
            "candidates": len(candidates),
            "accepted_score": round(accepted.score, 4) if accepted else "",
            "accepted_appearance": round(accepted.appearance_score, 4) if accepted else "",
            "accepted_distance_ratio": round(accepted.distance_ratio, 4) if accepted else "",
            "accepted_area_ratio": round(accepted.area_ratio, 4) if accepted else "",
            "accepted_iou": round(accepted.iou, 4) if accepted else "",
            "rejected_jumps": skier.rejected_jumps,
            "missed_frames": skier.missed_frames,
            "tracker_gap": skier.tracker_gap,
            "pending_reacquire_hits": skier.pending_reacquire_hits,
            "rejected_reacquire": skier.rejected_reacquire,
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
        }
    )


def model_class_filter(model: YOLO) -> List[int]:
    names = model.names
    if isinstance(names, dict):
        normalized = {int(class_id): str(name).lower() for class_id, name in names.items()}
    else:
        normalized = {class_id: str(name).lower() for class_id, name in enumerate(names)}

    skier_ids = [class_id for class_id, name in normalized.items() if name == SKIER_CLASS]
    if skier_ids:
        return skier_ids

    wanted = {PERSON_CLASS, *SKIER_EQUIPMENT_CLASSES}
    class_ids = [class_id for class_id, name in normalized.items() if name in wanted]
    return class_ids or YOLO_CLASSES


def has_custom_skier_class(model: YOLO) -> bool:
    names = model.names
    if isinstance(names, dict):
        return any(str(name).lower() == SKIER_CLASS for name in names.values())
    return any(str(name).lower() == SKIER_CLASS for name in names)


def effective_yolo_settings(model: YOLO, args: argparse.Namespace) -> Tuple[float, int]:
    if has_custom_skier_class(model):
        return min(args.conf, args.skier_conf), args.skier_imgsz
    return args.conf, args.imgsz


def run_yolo(model: YOLO, frame: np.ndarray, args: argparse.Namespace):
    conf, imgsz = effective_yolo_settings(model, args)
    return model.track(
        frame,
        persist=True,
        tracker=args.tracker,
        conf=conf,
        iou=args.iou,
        imgsz=imgsz,
        verbose=False,
        classes=model_class_filter(model),
    )


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    keyframe_boxes = load_keyframe_boxes(args.keyframes)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    skip_ranges = parse_skip_ranges(args.skip_ranges, video_fps)
    start_frame = parse_frame_or_time(args.start_at, video_fps) or 1
    end_frame = parse_frame_or_time(args.end_at, video_fps) or total_frames or 0
    if total_frames:
        start_frame = min(max(1, start_frame), total_frames)
        end_frame = min(max(start_frame, end_frame), total_frames)
    skipped_start = skipped_range_end(start_frame, skip_ranges)
    if skipped_start is not None:
        start_frame = skipped_start + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {start_frame} from {args.video}")

    names = model.names
    first_results = run_yolo(model, first_frame, args)
    first_detections = parse_detections(first_results[0], names)
    initial_box = interpolated_keyframe_box(start_frame, keyframe_boxes, first_frame.shape)
    if initial_box is None:
        initial_box = select_initial_box(first_frame, args, first_detections)

    initial_confidence = 1.0 if args.init_box or args.select_first_frame else 0.5
    skier = PrimarySkier(token_id=1, box=initial_box, confidence=initial_confidence)
    skier.update_appearance(first_frame, initial_box, alpha=1.0)
    skier._refresh_history()
    opencv_tracker, tracker_name = reset_opencv_tracker(first_frame, initial_box, args)

    writer = None
    actual_output_path = args.output
    csv_handle, csv_writer, actual_csv_path = open_csv_logger(args.log_csv)
    frame_number = start_frame
    processed_frames = 1
    skipped_frames = max(0, start_frame - 1)
    force_yolo_next = False
    frames_with_yolo_update = 0
    frames_with_opencv_update = 0
    frames_prediction_only = 0
    frames_with_manual_update = 0
    frames_with_keyframe_update = 0
    frames_lost = 0

    candidates = evaluate_candidates(first_frame, skier, first_detections, args)
    accepted = choose_candidate(candidates)
    if accepted is not None:
        skier.update_from_detection(
            first_frame,
            accepted.detection.xyxy,
            accepted.detection.confidence,
            accepted.detection.track_id,
            accepted.equipment_seen,
            args.smooth,
            appearance_alpha=0.10,
        )
        opencv_tracker, tracker_name = reset_opencv_tracker(first_frame, skier.box, args)
        frames_with_yolo_update += 1

    annotated = draw_skier_overlay(first_frame, skier, candidates, args, frame_number, tracker_name)
    if args.save:
        writer, actual_output_path = open_video_writer(args.output, cap, annotated.shape, playback_speed=args.playback_speed)
        writer.write(annotated)
    write_frame_log(csv_writer, frame_number, skier, candidates, accepted)
    if not args.no_display:
        cv2.imshow("Skier Tracking", annotated)
        cv2.waitKey(max(1, int(round((1000.0 / video_fps) / max(0.1, args.playback_speed)))))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_number += 1
        if end_frame and frame_number > end_frame:
            break
        if args.max_frames and processed_frames >= args.max_frames:
            break

        skip_end = skipped_range_end(frame_number, skip_ranges)
        if skip_end is not None:
            jump_to = skip_end + 1
            if end_frame and jump_to > end_frame:
                break
            skipped_frames += max(0, jump_to - frame_number)
            cap.set(cv2.CAP_PROP_POS_FRAMES, jump_to - 1)
            frame_number = jump_to - 1
            skier.prev_gray = None
            skier.flow_points = None
            opencv_tracker = None
            tracker_name = "skip-reset"
            force_yolo_next = True
            continue

        processed_frames += 1

        tracker_box = None
        flow_box = skier.estimate_optical_flow_box(frame)
        if opencv_tracker is not None:
            tracker_ok, xywh = opencv_tracker.update(frame)
            if tracker_ok:
                tracker_box = clamp_box(xywh_to_xyxy(xywh), frame.shape)

        manual_box = None
        if args.manual_every > 0 and not args.no_display and frame_number % args.manual_every == 0:
            manual_box = manually_correct_box(frame, skier, frame_number)
        keyframe_box = interpolated_keyframe_box(frame_number, keyframe_boxes, frame.shape)

        should_run_yolo = force_yolo_next or args.yolo_every <= 1 or frame_number % args.yolo_every == 0 or skier.missed_frames >= 12
        if manual_box is not None or keyframe_box is not None:
            detections = []
            candidates = []
        elif should_run_yolo:
            results = run_yolo(model, frame, args)
            detections = parse_detections(results[0], names)
            candidates = evaluate_candidates(frame, skier, detections, args, flow_box=flow_box)
            force_yolo_next = False
        else:
            detections = []
            candidates = []
        accepted = choose_candidate(candidates)
        if accepted is not None and not confirm_reacquisition(skier, accepted, args):
            accepted = None

        if manual_box is not None:
            skier.update_from_manual_box(frame, manual_box)
            opencv_tracker, tracker_name = reset_opencv_tracker(frame, skier.box, args)
            frames_with_manual_update += 1
        elif keyframe_box is not None:
            skier.update_from_manual_box(frame, keyframe_box, source="keyframe", smoothing=args.keyframe_blend)
            opencv_tracker, tracker_name = reset_opencv_tracker(frame, skier.box, args)
            frames_with_keyframe_update += 1
        elif accepted is not None:
            skier.update_from_detection(
                frame,
                accepted.detection.xyxy,
                accepted.detection.confidence,
                accepted.detection.track_id,
                accepted.equipment_seen,
                args.smooth,
                appearance_alpha=0.10,
            )
            opencv_tracker, tracker_name = reset_opencv_tracker(frame, skier.box, args)
            frames_with_yolo_update += 1
        elif flow_box is not None and skier.missed_frames < args.max_missed and skier.tracker_gap < args.max_tracker_gap:
            skier.update_from_tracker(flow_box, args.tracker_smooth, source="flow")
            frames_with_opencv_update += 1
        elif tracker_box is not None and skier.missed_frames < args.max_missed and skier.tracker_gap < args.max_tracker_gap:
            skier.update_from_tracker(tracker_box, args.tracker_smooth, source="opencv")
            frames_with_opencv_update += 1
        else:
            skier.mark_missed()
            frames_prediction_only += 1

        rejected_this_frame = sum(1 for candidate in candidates if not candidate.accepted)
        skier.rejected_jumps += rejected_this_frame
        if skier.missed_frames > args.max_missed:
            frames_lost += 1

        annotated = draw_skier_overlay(frame, skier, candidates, args, frame_number, tracker_name)

        if args.save:
            if writer is None:
                writer, actual_output_path = open_video_writer(args.output, cap, annotated.shape, playback_speed=args.playback_speed)
            writer.write(annotated)

        write_frame_log(csv_writer, frame_number, skier, candidates, accepted)

        if not args.no_display:
            cv2.imshow("Skier Tracking", annotated)
            preview_delay = max(1, int(round((1000.0 / video_fps) / max(0.1, args.playback_speed))))
            if cv2.waitKey(preview_delay) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to {actual_output_path}")
    if csv_handle is not None:
        csv_handle.close()
        print(f"Saved tracking diagnostics to {actual_csv_path}")
    if args.save or args.log_csv:
        effective_conf, effective_imgsz = effective_yolo_settings(model, args)
        summary = {
            "video": args.video,
            "model": args.model,
            "detector_classes": [model.names[class_id] for class_id in model_class_filter(model)],
            "custom_skier_model": has_custom_skier_class(model),
            "effective_conf": effective_conf,
            "effective_imgsz": effective_imgsz,
            "keyframes_loaded": len(keyframe_boxes),
            "tracker": args.tracker,
            "opencv_tracker": tracker_name,
            "yolo_every": args.yolo_every,
            "playback_speed": args.playback_speed,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "skip_ranges": skip_ranges,
            "skipped_frames": skipped_frames,
            "last_source_frame": frame_number,
            "frames_processed": processed_frames,
            "frames_with_yolo_update": frames_with_yolo_update,
            "frames_with_manual_update": frames_with_manual_update,
            "frames_with_keyframe_update": frames_with_keyframe_update,
            "frames_with_opencv_update": frames_with_opencv_update,
            "frames_prediction_only": frames_prediction_only,
            "max_tracker_gap": args.max_tracker_gap,
            "frames_lost_after_max_missed": frames_lost,
            "rejected_candidate_jumps": skier.rejected_jumps,
            "rejected_reacquire_candidates": skier.rejected_reacquire,
            "final_state": skier.state,
            "final_box": [round(float(value), 2) for value in skier.box],
            "output_video": actual_output_path if args.save else None,
            "diagnostics_csv": actual_csv_path,
            "recommendations": [
                "Use --select-first-frame for the most reliable identity anchor.",
                "If generic YOLO still picks the wrong subject, use --manual-every 30 for correction keyframes.",
                "Use --manual-every 1 for true manual frame-by-frame tracking.",
                "The default run now uses frame-by-frame YOLO with --yolo-every 1.",
                "Custom skier models automatically use --skier-conf and --skier-imgsz so low-confidence turn detections can still be evaluated.",
                "If the box sticks to the wrong region too long, reduce --max-missed or --max-jump.",
                "If reacquisition is too conservative, lower --appearance-threshold or raise --max-jump slightly.",
                "The next major quality jump is more corrected skier labels and a longer training run.",
            ],
        }
        write_summary(args.summary, summary)
        print(f"Saved run summary to {args.summary}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
