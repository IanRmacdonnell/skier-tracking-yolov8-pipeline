import argparse
import os
from pathlib import Path

LOCAL_RUNTIME_DIR = Path(".runtime")
ULTRALYTICS_CONFIG_DIR = LOCAL_RUNTIME_DIR / "ultralytics"
MATPLOTLIB_CONFIG_DIR = LOCAL_RUNTIME_DIR / "matplotlib"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR.resolve()))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a one-class skier YOLO detector.")
    parser.add_argument("--model", default="yolov8s.pt", help="Starting YOLO model.")
    parser.add_argument("--data", default="dataset/skier_clean/data.yaml", help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=4, help="Training batch size.")
    parser.add_argument("--device", default="", help="Device, for example 0, cpu, or empty for auto.")
    parser.add_argument("--project", default="runs/skier", help="Training output project directory.")
    parser.add_argument("--name", default="train", help="Training run name.")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers. 0 is safest on Windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    train_args = {
        "data": args.data,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
    }
    if args.device:
        train_args["device"] = args.device
    results = model.train(**train_args)
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"
    print(f"Training results: {save_dir}")
    print(f"Best weights: {best_weights}")
    print(f"Last weights: {last_weights}")


if __name__ == "__main__":
    main()
