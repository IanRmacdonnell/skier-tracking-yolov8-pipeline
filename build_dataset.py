import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2


@dataclass(frozen=True)
class Box:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a one-class YOLO skier dataset from keyframe boxes.")
    parser.add_argument("--video", default="videos/ski.mp4", help="Input video path.")
    parser.add_argument("--keyframes", default="annotations/keyframes.csv", help="CSV from label_skier.py.")
    parser.add_argument("--output", default="dataset/skier", help="YOLO dataset output directory.")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame between labeled keyframes.")
    parser.add_argument("--val-every", type=int, default=5, help="Put every Nth exported frame in validation.")
    parser.add_argument("--padding", type=float, default=0.0, help="Optional box padding ratio, e.g. 0.10.")
    parser.add_argument("--keyframes-only", action="store_true", help="Use only manually labeled frames; no interpolation.")
    parser.add_argument("--overwrite", action="store_true", help="Delete and rebuild the output dataset directory.")
    return parser.parse_args()


def load_keyframes(path: Path) -> List[Box]:
    if not path.exists():
        raise FileNotFoundError(f"Missing keyframes CSV: {path}")
    boxes = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            boxes.append(
                Box(
                    frame=int(row["frame"]),
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                )
            )
    boxes.sort(key=lambda box: box.frame)
    if len(boxes) < 1:
        raise ValueError("No keyframes found.")
    return boxes


def interpolate_box(frame_number: int, keyframes: List[Box]) -> Optional[Box]:
    if frame_number < keyframes[0].frame or frame_number > keyframes[-1].frame:
        return None

    exact = {box.frame: box for box in keyframes}
    if frame_number in exact:
        return exact[frame_number]

    previous = None
    following = None
    for box in keyframes:
        if box.frame < frame_number:
            previous = box
        elif box.frame > frame_number:
            following = box
            break

    if previous is None or following is None:
        return None

    span = following.frame - previous.frame
    if span <= 0:
        return previous
    t = (frame_number - previous.frame) / span
    return Box(
        frame=frame_number,
        x1=previous.x1 + ((following.x1 - previous.x1) * t),
        y1=previous.y1 + ((following.y1 - previous.y1) * t),
        x2=previous.x2 + ((following.x2 - previous.x2) * t),
        y2=previous.y2 + ((following.y2 - previous.y2) * t),
    )


def pad_and_clamp(box: Box, width: int, height: int, padding: float) -> Box:
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    pad_x = max(0.0, (x2 - x1) * padding)
    pad_y = max(0.0, (y2 - y1) * padding)
    return Box(
        frame=box.frame,
        x1=max(0.0, x1 - pad_x),
        y1=max(0.0, y1 - pad_y),
        x2=min(float(width - 1), x2 + pad_x),
        y2=min(float(height - 1), y2 + pad_y),
    )


def yolo_label(box: Box, width: int, height: int) -> str:
    box_width = max(1.0, box.x2 - box.x1)
    box_height = max(1.0, box.y2 - box.y1)
    x_center = box.x1 + (box_width / 2.0)
    y_center = box.y1 + (box_height / 2.0)
    return "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
        x_center / width,
        y_center / height,
        box_width / width,
        box_height / height,
    )


def next_available_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available output directory near {path}")


def make_dirs(output: Path, overwrite: bool) -> tuple[Path, Dict[str, Path]]:
    if output.exists() and overwrite:
        try:
            shutil.rmtree(output)
        except PermissionError:
            print(f"{output} is locked; writing to a new suffixed dataset directory instead.")
            output = next_available_dir(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"{output} already exists. Use --overwrite to rebuild it.")

    paths = {
        "train_images": output / "images" / "train",
        "val_images": output / "images" / "val",
        "train_labels": output / "labels" / "train",
        "val_labels": output / "labels" / "val",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return output, paths


def write_data_yaml(output: Path) -> None:
    (output / "data.yaml").write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: skier\n",
        encoding="utf-8",
    )


def selected_frames(keyframes: List[Box], frame_step: int, keyframes_only: bool) -> List[int]:
    if keyframes_only:
        return [box.frame for box in keyframes]
    first = keyframes[0].frame
    last = keyframes[-1].frame
    return list(range(first, last + 1, max(1, frame_step)))


def main() -> None:
    args = parse_args()
    keyframes = load_keyframes(Path(args.keyframes))
    output = Path(args.output)
    output, paths = make_dirs(output, args.overwrite)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    frames = selected_frames(keyframes, args.frame_step, args.keyframes_only)
    exported = 0
    train_count = 0
    val_count = 0

    try:
        for index, frame_number in enumerate(frames, start=1):
            box = interpolate_box(frame_number, keyframes)
            if box is None:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ok, frame = cap.read()
            if not ok:
                print(f"Could not read frame {frame_number}; skipping.")
                continue

            height, width = frame.shape[:2]
            box = pad_and_clamp(box, width, height, args.padding)
            split = "val" if args.val_every > 0 and index % args.val_every == 0 else "train"
            stem = f"ski_{frame_number:06d}"

            image_path = paths[f"{split}_images"] / f"{stem}.jpg"
            label_path = paths[f"{split}_labels"] / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame)
            label_path.write_text(yolo_label(box, width, height), encoding="utf-8")

            exported += 1
            if split == "val":
                val_count += 1
            else:
                train_count += 1
    finally:
        cap.release()

    write_data_yaml(output)
    print(f"Exported {exported} labeled frames to {output}")
    print(f"Train: {train_count}, val: {val_count}")
    print(f"Data YAML: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
