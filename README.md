# Skier Tracking YOLOv8 Pipeline

**YOLOv8 skier-tracking pipeline for learning computer vision and data engineering: labeling video, training detection models, tracking frames, and exporting quality metrics.**

This project tracks one skier through video using YOLOv8, OpenCV tracking, manual keyframes, custom one-class training, and a data pipeline that turns frame-level model decisions into structured analytics outputs.

## At a Glance

| Area | Details |
| --- | --- |
| Project type | Computer vision and data-engineering prep project |
| Main objective | Learn YOLOv8 and the surrounding AI data workflow before a summer internship |
| Core workflow | Raw video -> labels -> YOLO dataset -> trained detector -> tracked frames -> diagnostics -> quality metrics -> review queue |
| Tech stack | Python, YOLOv8/Ultralytics, OpenCV, NumPy, CSV/JSON, optional Polars/Parquet |
| Best portfolio signal | Connects model output to practical data engineering, quality checks, and human review |

## About / Examples

This project is built around the kind of work that happens around an AI model, not just inside the model.

Example situations it handles:

- Label keyframes from ski footage and convert them into a YOLO one-class `skier` dataset.
- Train a custom YOLOv8 detector and use it to track a skier through a video.
- Bridge weak detection moments with OpenCV tracking and optical flow.
- Export per-frame diagnostics so tracking decisions can be reviewed later.
- Convert tracking logs into structured tables for runs, frames, detections, tracks, metrics, and review queues.
- Identify hard frames that should be relabeled, creating a feedback loop for improving the dataset.

## Why I Built It

The goal was not just to make a detector. The goal was to understand the full workflow around video-derived AI data:

```text
raw ski video -> labels -> YOLO dataset -> trained detector -> tracked frames -> diagnostics -> quality metrics -> review queue
```

That makes the project useful internship prep because it connects machine learning output to the kind of data engineering work needed for reliable AI video products.

## What It Does

- Detects and tracks a selected skier frame by frame.
- Uses YOLOv8 for detection and OpenCV tracking/optical flow to bridge short gaps.
- Supports manual first-frame selection and keyframe corrections.
- Builds a custom one-class `skier` YOLO dataset from labeled frames.
- Trains a custom YOLOv8 skier detector.
- Saves annotated videos, per-frame logs, and JSON summaries.
- Converts tracking logs into structured tables for downstream analysis.
- Produces quality metrics and a label-review queue for weak tracking moments.
- Supports video review controls like playback speed, start/end clips, and skipped time ranges.

## Skills Demonstrated

- Computer vision fundamentals: object detection, tracking, confidence thresholds, identity drift, and frame-level debugging.
- YOLOv8 workflow: labeling, dataset building, training, validation, and inference.
- Data engineering: converting model output into structured tables, partitions, quality metrics, and review queues.
- MLOps-style iteration: using failure cases to generate new labeling targets.
- Python CLI design with reusable scripts and configurable run parameters.
- Data quality thinking: lock rate, lost frames, prediction-only frames, flow bridges, and human review priorities.
- Practical internship prep: understanding how raw video becomes product data, not just model predictions.

## Tech Stack

- Python
- YOLOv8 / Ultralytics
- OpenCV
- NumPy
- Polars for optional Parquet output
- CSV/JSON diagnostics

## Project Structure

```text
.
|-- main.py                       # Detection/tracking CLI
|-- tracker_math.py               # Geometry, IoU, smoothing helpers
|-- tracking_io.py                # Video/log/summary output helpers
|-- tracking_data_pipeline.py     # Converts logs into structured data tables
|-- label_skier.py                # Manual labeling tool
|-- build_dataset.py              # YOLO dataset builder
|-- train_skier.py                # Ultralytics training wrapper
|-- audit_labels.py               # Label quality audit helper
|-- clean_keyframes.py            # Label cleanup helper
|-- annotations/
|   |-- hard_moments_to_label.csv # Frames worth reviewing next
|   `-- sample_keyframes.csv      # Small label format example
`-- requirements.txt
```

Large generated artifacts are intentionally excluded from GitHub:

- `videos/`
- `dataset/`
- `runs/`
- `output/`
- `data/processed/`
- `*.pt` model weights

## Setup

Install Python 3.10+.

```bash
pip install -r requirements.txt
```

You will need your own video file and YOLO weights. The original local project used `videos/ski.mp4`, but videos and trained weights are not included in this repository to keep it lightweight and GitHub-friendly.

## Run Tracking

Preview tracking with first-frame selection:

```bash
python main.py --select-first-frame
```

Save video, frame diagnostics, and a JSON summary:

```bash
python main.py --select-first-frame --save --log-csv output/tracking_log.csv --summary output/tracking_summary.json
```

Use a trained skier model:

```bash
python main.py --model runs/detect/runs/skier/skier_clean_3/weights/best.pt --select-first-frame --save
```

Save a faster review video:

```bash
python main.py --model runs/detect/runs/skier/skier_clean_3/weights/best.pt --select-first-frame --save --playback-speed 2.0
```

Skip sections while preserving original frame numbering:

```bash
python main.py --model runs/detect/runs/skier/skier_clean_3/weights/best.pt --select-first-frame --save --start-at 0:20 --end-at 1:45 --skip-ranges 0:45-0:55,1:10-1:18
```

## Build Training Data

Label keyframes:

```bash
python label_skier.py --video videos/ski.mp4 --every 15 --output annotations/keyframes.csv
```

Label known hard moments:

```bash
python label_skier.py --video videos/ski.mp4 --frames-file annotations/hard_moments_to_label.csv --output annotations/keyframes.csv --redo
```

Build a YOLO dataset:

```bash
python build_dataset.py --video videos/ski.mp4 --keyframes annotations/keyframes_clean.csv --output dataset/skier_clean --frame-step 2 --val-every 5 --padding 0.10 --overwrite
```

Train:

```bash
python train_skier.py --model yolov8s.pt --data dataset/skier_clean/data.yaml --imgsz 640 --epochs 50 --batch 4 --name skier_clean_long --workers 0
```

## Data Pipeline

After a tracking run:

```bash
python tracking_data_pipeline.py --log output/tracking_log.csv --summary output/tracking_summary.json --label-file annotations/keyframes_clean.csv
```

The pipeline creates structured tables for:

- runs
- model_runs
- label_versions
- frames
- detections
- tracks
- tracking_quality_metrics
- review_queue

Useful quality metrics include lock rate, lost frames, prediction-only frames, flow bridge frames, average appearance score, and longest non-locked streak.

## What I Would Highlight

Resume version:

> Built a YOLOv8 skier-tracking pipeline that labels ski footage, trains a one-class detector, tracks skier identity frame by frame, logs model decisions, and converts tracking output into structured quality metrics and review queues.

Internship version:

> Used a computer vision project to practice data engineering around AI systems: model output logging, table design, data quality metrics, Parquet-ready exports, and human-in-the-loop label review.

Technical version:

> Implemented a feedback loop where weak tracking moments become targeted relabeling candidates, improving the training set instead of only tuning inference thresholds.

## Next Improvements

- Move processed outputs to cloud/object storage style paths.
- Add a small SQL or DuckDB notebook for querying tracking quality.
- Add automated tests for time parsing, skip ranges, and data pipeline outputs.
- Add a lightweight demo video or GIF if licensing allows.
- Track multiple skiers with IDs once the single-skier pipeline is stable.
