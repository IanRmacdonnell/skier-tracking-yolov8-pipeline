import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import cv2


FIELDNAMES = ["frame", "x1", "y1", "x2", "y2", "width", "height"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label skier keyframes from a video.")
    parser.add_argument("--video", default="videos/ski.mp4", help="Input video path.")
    parser.add_argument("--every", type=int, default=15, help="Label every Nth frame.")
    parser.add_argument("--start", type=int, default=1, help="First 1-based frame to label.")
    parser.add_argument("--end", type=int, default=0, help="Last 1-based frame to label. Default is video end.")
    parser.add_argument("--frames", default="", help="Comma-separated 1-based frames to label instead of using --every.")
    parser.add_argument("--frames-file", default="", help="CSV file with a frame column listing exact frames to label.")
    parser.add_argument("--output", default="annotations/keyframes.csv", help="CSV output path.")
    parser.add_argument("--redo", action="store_true", help="Relabel frames already present in the CSV.")
    return parser.parse_args()


def load_existing(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {int(row["frame"]): row for row in csv.DictReader(handle)}


def save_annotations(path: Path, rows: Dict[int, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for frame_number in sorted(rows):
            writer.writerow(rows[frame_number])


def parse_frame_targets(args: argparse.Namespace, total_frames: int) -> List[int]:
    frames = set()
    if args.frames:
        for item in args.frames.split(","):
            item = item.strip()
            if item:
                frames.add(int(item))

    if args.frames_file:
        with Path(args.frames_file).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                frames.add(int(row["frame"]))

    if frames:
        return sorted(frame for frame in frames if 1 <= frame <= total_frames)

    end_frame = args.end if args.end > 0 else total_frames
    end_frame = min(end_frame, total_frames)
    start_frame = max(1, args.start)
    return list(range(start_frame, end_frame + 1, args.every))


def read_frame(cap: cv2.VideoCapture, frame_number: int) -> Optional[object]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = cap.read()
    return frame if ok else None


def draw_prompt(frame, frame_number: int, total_frames: int, existing: Optional[Dict[str, str]]):
    preview = frame.copy()
    if existing:
        x1 = int(float(existing["x1"]))
        y1 = int(float(existing["y1"]))
        x2 = int(float(existing["x2"]))
        y2 = int(float(existing["y2"]))
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 180, 255), 2)
    text = f"Frame {frame_number}/{total_frames}: draw skier box. ENTER/SPACE saves, C skips."
    cv2.putText(preview, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
    return preview


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    annotations = load_existing(output_path)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_targets = parse_frame_targets(args, total_frames)

    print(f"Labeling {args.video}")
    print(f"Frames to label: {len(frame_targets)}")
    print(f"Saving keyframes to {output_path}")

    try:
        for frame_number in frame_targets:
            if frame_number in annotations and not args.redo:
                continue

            frame = read_frame(cap, frame_number)
            if frame is None:
                print(f"Could not read frame {frame_number}; skipping.")
                continue

            preview = draw_prompt(frame, frame_number, total_frames, annotations.get(frame_number))
            roi = cv2.selectROI("Label Skier Keyframe", preview, fromCenter=False, showCrosshair=True)
            if roi[2] <= 0 or roi[3] <= 0:
                print(f"Skipped frame {frame_number}")
                continue

            x, y, w, h = roi
            row = {
                "frame": frame_number,
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + w),
                "y2": int(y + h),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            }
            annotations[frame_number] = row
            save_annotations(output_path, annotations)
            print(f"Saved frame {frame_number}: {row['x1']},{row['y1']},{row['x2']},{row['y2']}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    save_annotations(output_path, annotations)
    print(f"Done. Labeled {len(annotations)} keyframes in {output_path}")


if __name__ == "__main__":
    main()
