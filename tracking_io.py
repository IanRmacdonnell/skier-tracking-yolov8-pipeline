import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available output path near {path}")


def prepare_output_path(path: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            output_path.unlink()
        except PermissionError:
            output_path = next_available_path(output_path)
    return output_path


def open_video_writer(path: str, cap: cv2.VideoCapture, frame_shape: Tuple[int, int, int], playback_speed: float = 1.0) -> Tuple[cv2.VideoWriter, str]:
    output_path = prepare_output_path(path)
    playback_speed = max(0.1, float(playback_speed))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) * playback_speed
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height)), str(output_path)


def open_csv_logger(path: str):
    if not path:
        return None, None, None
    csv_path = prepare_output_path(path)
    handle = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "frame",
            "state",
            "source",
            "candidates",
            "accepted_score",
            "accepted_appearance",
            "accepted_distance_ratio",
            "accepted_area_ratio",
            "accepted_iou",
            "rejected_jumps",
            "missed_frames",
            "tracker_gap",
            "pending_reacquire_hits",
            "rejected_reacquire",
            "x1",
            "y1",
            "x2",
            "y2",
        ],
    )
    writer.writeheader()
    return handle, writer, str(csv_path)


def write_summary(path: str, summary: Dict[str, object]) -> None:
    summary_path = prepare_output_path(path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if str(summary_path) != path:
        print(f"Summary path was locked; saved to {summary_path}")
