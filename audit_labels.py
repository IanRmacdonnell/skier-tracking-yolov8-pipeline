import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Keyframe:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0], dtype=float)

    @property
    def size(self) -> np.ndarray:
        return np.array([max(1.0, self.x2 - self.x1), max(1.0, self.y2 - self.y1)], dtype=float)

    @property
    def area(self) -> float:
        width, height = self.size
        return float(width * height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit skier keyframe labels and create visual review sheets.")
    parser.add_argument("--video", default="videos/ski.mp4", help="Input video path.")
    parser.add_argument("--keyframes", default="annotations/keyframes.csv", help="Keyframe CSV.")
    parser.add_argument("--output", default="output/label_audit", help="Audit output directory.")
    parser.add_argument("--top", type=int, default=24, help="Number of sharp-change frames to report.")
    parser.add_argument("--sheet-cols", type=int, default=4, help="Images per contact-sheet row.")
    return parser.parse_args()


def load_keyframes(path: Path) -> List[Keyframe]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Keyframe(
                    frame=int(row["frame"]),
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                )
            )
    rows.sort(key=lambda item: item.frame)
    return rows


def score_segments(keyframes: List[Keyframe]) -> List[dict]:
    segments = []
    previous_motion = None
    for prev, curr in zip(keyframes, keyframes[1:]):
        frame_gap = max(1, curr.frame - prev.frame)
        motion = (curr.center - prev.center) / frame_gap
        speed = float(np.linalg.norm(motion))
        size_ratio = float(max(curr.area, prev.area) / max(1.0, min(curr.area, prev.area)))
        turn_score = 0.0
        if previous_motion is not None:
            denom = max(1e-6, np.linalg.norm(previous_motion) * np.linalg.norm(motion))
            cosine = float(np.clip(np.dot(previous_motion, motion) / denom, -1.0, 1.0))
            turn_score = float((1.0 - cosine) / 2.0)
        risk = speed + (18.0 * turn_score) + (6.0 * max(0.0, size_ratio - 1.0))
        segments.append(
            {
                "start_frame": prev.frame,
                "end_frame": curr.frame,
                "speed_px_per_frame": round(speed, 3),
                "turn_score": round(turn_score, 3),
                "size_ratio": round(size_ratio, 3),
                "risk": round(risk, 3),
            }
        )
        previous_motion = motion
    segments.sort(key=lambda item: item["risk"], reverse=True)
    return segments


def read_frame(cap: cv2.VideoCapture, frame_number: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = cap.read()
    return frame if ok else None


def draw_keyframe(frame, keyframe: Keyframe):
    image = frame.copy()
    x1, y1, x2, y2 = [int(round(value)) for value in (keyframe.x1, keyframe.y1, keyframe.x2, keyframe.y2)]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(image, f"frame {keyframe.frame}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return image


def make_contact_sheet(images: List[np.ndarray], cols: int) -> np.ndarray:
    if not images:
        raise ValueError("No images for contact sheet.")
    thumb_w = 420
    thumb_h = 236
    thumbs = [cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA) for image in images]
    rows = []
    for idx in range(0, len(thumbs), cols):
        row = thumbs[idx : idx + cols]
        while len(row) < cols:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    keyframes = load_keyframes(Path(args.keyframes))
    if len(keyframes) < 2:
        raise ValueError("Need at least two keyframes to audit.")

    segments = score_segments(keyframes)
    report = {
        "keyframe_count": len(keyframes),
        "first_frame": keyframes[0].frame,
        "last_frame": keyframes[-1].frame,
        "top_risk_segments": segments[: args.top],
    }
    (output / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    keyframe_by_number = {item.frame: item for item in keyframes}

    review_frames = []
    for segment in segments[: args.top]:
        review_frames.append(segment["start_frame"])
        review_frames.append(segment["end_frame"])
    review_frames = sorted(set(review_frames))

    images = []
    try:
        for frame_number in review_frames:
            frame = read_frame(cap, frame_number)
            if frame is None or frame_number not in keyframe_by_number:
                continue
            images.append(draw_keyframe(frame, keyframe_by_number[frame_number]))
    finally:
        cap.release()

    if images:
        sheet = make_contact_sheet(images, max(1, args.sheet_cols))
        cv2.imwrite(str(output / "review_sheet.jpg"), sheet)

    print(f"Wrote audit report to {output / 'audit_report.json'}")
    if images:
        print(f"Wrote review sheet to {output / 'review_sheet.jpg'}")


if __name__ == "__main__":
    main()
