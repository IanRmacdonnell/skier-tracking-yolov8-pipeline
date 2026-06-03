import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


FRAME_FIELDS = [
    "run_id",
    "frame",
    "state",
    "source",
    "candidates",
    "accepted_score",
    "accepted_appearance",
    "accepted_distance_ratio",
    "accepted_area_ratio",
    "accepted_iou",
    "missed_frames",
    "tracker_gap",
    "pending_reacquire_hits",
    "x1",
    "y1",
    "x2",
    "y2",
]

DETECTION_FIELDS = [
    "run_id",
    "frame",
    "detection_id",
    "accepted_score",
    "accepted_appearance",
    "accepted_distance_ratio",
    "accepted_area_ratio",
    "accepted_iou",
    "x1",
    "y1",
    "x2",
    "y2",
]

TRACK_FIELDS = [
    "run_id",
    "track_id",
    "frame_start",
    "frame_end",
    "frames",
    "lock_rate",
    "lost_frames",
    "prediction_only_frames",
    "flow_bridge_frames",
    "avg_accepted_score",
    "avg_appearance",
    "longest_non_locked_streak",
    "longest_prediction_streak",
]

RUN_FIELDS = [
    "run_id",
    "created_at",
    "video",
    "model",
    "frames_processed",
    "detector_classes",
    "effective_conf",
    "effective_imgsz",
    "log_path",
    "summary_path",
]

MODEL_RUN_FIELDS = [
    "run_id",
    "model",
    "detector_classes",
    "custom_skier_model",
    "effective_conf",
    "effective_imgsz",
    "tracker",
    "opencv_tracker",
    "yolo_every",
]

LABEL_VERSION_FIELDS = [
    "run_id",
    "label_file",
    "label_version_id",
    "keyframes_loaded",
]

METRIC_FIELDS = ["run_id", "metric", "value"]

REVIEW_FIELDS = [
    "run_id",
    "frame",
    "priority",
    "reason",
    "state",
    "source",
    "accepted_score",
    "accepted_appearance",
    "missed_frames",
    "tracker_gap",
    "x1",
    "y1",
    "x2",
    "y2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert tracker logs into data-engineering tables and quality metrics.")
    parser.add_argument("--log", required=True, help="Tracking diagnostics CSV from main.py.")
    parser.add_argument("--summary", default="", help="Optional tracking summary JSON from main.py.")
    parser.add_argument("--output", default="data/processed/tracks", help="Output dataset root.")
    parser.add_argument("--run-id", default="", help="Stable run id. Default is derived from log path and contents.")
    parser.add_argument("--label-file", default="annotations/keyframes_clean.csv", help="Label file used for this model/run.")
    parser.add_argument("--review-output", default="annotations/review_queue.csv", help="Human review queue CSV path.")
    parser.add_argument("--format", choices=["csv", "parquet", "both"], default="both", help="Table output format.")
    parser.add_argument("--min-score", type=float, default=1.10, help="Accepted score below this is review-worthy.")
    parser.add_argument("--min-appearance", type=float, default=0.90, help="Appearance below this is review-worthy.")
    parser.add_argument("--max-tracker-gap", type=int, default=2, help="Tracker gap above this is review-worthy.")
    return parser.parse_args()


def read_json(path: str) -> Dict[str, object]:
    if not path:
        return {}
    summary_path = Path(path)
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def stable_run_id(log_path: Path, rows: List[Dict[str, str]]) -> str:
    seed = f"{log_path.resolve()}:{len(rows)}:{rows[0].get('frame', '') if rows else ''}:{rows[-1].get('frame', '') if rows else ''}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def label_version_id(path: str) -> str:
    label_path = Path(path)
    if not label_path.exists():
        return ""
    return hashlib.sha1(label_path.read_bytes()).hexdigest()[:12]


def average(values: Iterable[Optional[float]]) -> float:
    clean = [value for value in values if value is not None]
    return round(sum(clean) / len(clean), 6) if clean else 0.0


def longest_streak(rows: List[Dict[str, object]], predicate) -> int:
    longest = 0
    current = 0
    for row in rows:
        if predicate(row):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def normalize_frames(rows: List[Dict[str, str]], run_id: str) -> List[Dict[str, object]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "run_id": run_id,
                "frame": safe_int(row.get("frame")),
                "state": row.get("state", ""),
                "source": row.get("source", ""),
                "candidates": safe_int(row.get("candidates")),
                "accepted_score": safe_float(row.get("accepted_score")),
                "accepted_appearance": safe_float(row.get("accepted_appearance")),
                "accepted_distance_ratio": safe_float(row.get("accepted_distance_ratio")),
                "accepted_area_ratio": safe_float(row.get("accepted_area_ratio")),
                "accepted_iou": safe_float(row.get("accepted_iou")),
                "missed_frames": safe_int(row.get("missed_frames")),
                "tracker_gap": safe_int(row.get("tracker_gap")),
                "pending_reacquire_hits": safe_int(row.get("pending_reacquire_hits")),
                "x1": safe_float(row.get("x1"), 0.0),
                "y1": safe_float(row.get("y1"), 0.0),
                "x2": safe_float(row.get("x2"), 0.0),
                "y2": safe_float(row.get("y2"), 0.0),
            }
        )
    return normalized


def build_detections(frames: List[Dict[str, object]]) -> List[Dict[str, object]]:
    detections = []
    for row in frames:
        if row["accepted_score"] is None:
            continue
        frame = int(row["frame"])
        detections.append(
            {
                "run_id": row["run_id"],
                "frame": frame,
                "detection_id": f"{row['run_id']}_{frame:06d}_accepted",
                "accepted_score": row["accepted_score"],
                "accepted_appearance": row["accepted_appearance"],
                "accepted_distance_ratio": row["accepted_distance_ratio"],
                "accepted_area_ratio": row["accepted_area_ratio"],
                "accepted_iou": row["accepted_iou"],
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
            }
        )
    return detections


def build_metrics(frames: List[Dict[str, object]], run_id: str) -> Dict[str, float]:
    total = max(1, len(frames))
    locked = sum(1 for row in frames if row["state"] == "LOCKED")
    lost = sum(1 for row in frames if row["state"] == "LOST")
    prediction = sum(1 for row in frames if row["source"] == "prediction")
    flow = sum(1 for row in frames if row["source"] == "flow")
    yolo = sum(1 for row in frames if row["source"] == "yolo")
    return {
        "frames": len(frames),
        "lock_rate": round(locked / total, 6),
        "lost_frames": lost,
        "prediction_only_frames": prediction,
        "flow_bridge_frames": flow,
        "yolo_update_frames": yolo,
        "avg_accepted_score": average(row["accepted_score"] for row in frames),
        "avg_appearance": average(row["accepted_appearance"] for row in frames),
        "longest_non_locked_streak": longest_streak(frames, lambda row: row["state"] != "LOCKED"),
        "longest_prediction_streak": longest_streak(frames, lambda row: row["source"] == "prediction"),
        "max_missed_frames": max((int(row["missed_frames"]) for row in frames), default=0),
        "max_tracker_gap": max((int(row["tracker_gap"]) for row in frames), default=0),
    }


def build_review_queue(frames: List[Dict[str, object]], args: argparse.Namespace) -> List[Dict[str, object]]:
    queue = []
    for row in frames:
        reasons = []
        score = row["accepted_score"]
        appearance = row["accepted_appearance"]
        if row["state"] != "LOCKED":
            reasons.append("not_locked")
        if row["source"] in {"prediction", "flow"}:
            reasons.append(f"{row['source']}_bridge")
        if score is not None and score < args.min_score:
            reasons.append("low_score")
        if appearance is not None and appearance < args.min_appearance:
            reasons.append("low_appearance")
        if int(row["tracker_gap"]) > args.max_tracker_gap:
            reasons.append("tracker_gap")
        if not reasons:
            continue

        priority = 3 if row["state"] == "LOST" or row["source"] == "prediction" else 2 if row["state"] != "LOCKED" else 1
        queue.append(
            {
                "run_id": row["run_id"],
                "frame": row["frame"],
                "priority": priority,
                "reason": "|".join(reasons),
                "state": row["state"],
                "source": row["source"],
                "accepted_score": row["accepted_score"],
                "accepted_appearance": row["accepted_appearance"],
                "missed_frames": row["missed_frames"],
                "tracker_gap": row["tracker_gap"],
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
            }
        )
    return queue


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_table(path: Path, rows: List[Dict[str, object]], fieldnames: List[str], output_format: str) -> None:
    if output_format in {"csv", "both"}:
        write_csv(path.with_suffix(".csv"), rows, fieldnames)
    if output_format in {"parquet", "both"}:
        try:
            import polars as pl

            path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(rows, schema=fieldnames).write_parquet(path.with_suffix(".parquet"))
        except ImportError:
            if output_format == "parquet":
                raise RuntimeError("Parquet output requires polars. Install it or use --format csv.")


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    rows = read_csv(log_path)
    run_id = args.run_id or stable_run_id(log_path, rows)
    summary = read_json(args.summary)
    frames = normalize_frames(rows, run_id)
    detections = build_detections(frames)
    metrics = build_metrics(frames, run_id)
    review_queue = build_review_queue(frames, args)

    created_at = datetime.now(timezone.utc).isoformat()
    video = str(summary.get("video", ""))
    model = str(summary.get("model", ""))
    video_name = Path(video).stem if video else "unknown_video"
    run_root = Path(args.output) / f"date={created_at[:10]}" / f"video={video_name}" / f"run_id={run_id}"

    runs = [
        {
            "run_id": run_id,
            "created_at": created_at,
            "video": video,
            "model": model,
            "frames_processed": int(summary.get("frames_processed", len(frames))),
            "detector_classes": "|".join(str(item) for item in summary.get("detector_classes", [])),
            "effective_conf": summary.get("effective_conf", ""),
            "effective_imgsz": summary.get("effective_imgsz", ""),
            "log_path": str(log_path),
            "summary_path": args.summary,
        }
    ]
    model_runs = [
        {
            "run_id": run_id,
            "model": model,
            "detector_classes": "|".join(str(item) for item in summary.get("detector_classes", [])),
            "custom_skier_model": summary.get("custom_skier_model", ""),
            "effective_conf": summary.get("effective_conf", ""),
            "effective_imgsz": summary.get("effective_imgsz", ""),
            "tracker": summary.get("tracker", ""),
            "opencv_tracker": summary.get("opencv_tracker", ""),
            "yolo_every": summary.get("yolo_every", ""),
        }
    ]
    label_versions = [
        {
            "run_id": run_id,
            "label_file": args.label_file,
            "label_version_id": label_version_id(args.label_file),
            "keyframes_loaded": summary.get("keyframes_loaded", ""),
        }
    ]
    tracks = [
        {
            "run_id": run_id,
            "track_id": "primary_skier",
            "frame_start": frames[0]["frame"] if frames else 0,
            "frame_end": frames[-1]["frame"] if frames else 0,
            **metrics,
        }
    ]
    metric_rows = [{"run_id": run_id, "metric": key, "value": value} for key, value in metrics.items()]

    write_table(run_root / "runs", runs, RUN_FIELDS, args.format)
    write_table(run_root / "model_runs", model_runs, MODEL_RUN_FIELDS, args.format)
    write_table(run_root / "label_versions", label_versions, LABEL_VERSION_FIELDS, args.format)
    write_table(run_root / "frames", frames, FRAME_FIELDS, args.format)
    write_table(run_root / "detections", detections, DETECTION_FIELDS, args.format)
    write_table(run_root / "tracks", tracks, TRACK_FIELDS, args.format)
    write_table(run_root / "tracking_quality_metrics", metric_rows, METRIC_FIELDS, args.format)
    write_table(run_root / "review_queue", review_queue, REVIEW_FIELDS, args.format)

    review_path = Path(args.review_output)
    write_csv(review_path, review_queue, REVIEW_FIELDS)
    (run_root / "quality_summary.json").write_text(json.dumps({"run_id": run_id, **metrics, "review_frames": len(review_queue)}, indent=2), encoding="utf-8")

    print(f"Run id: {run_id}")
    print(f"Dataset written to: {run_root}")
    print(f"Review queue written to: {review_path}")
    print(json.dumps({"lock_rate": metrics["lock_rate"], "lost_frames": metrics["lost_frames"], "review_frames": len(review_queue)}, indent=2))


if __name__ == "__main__":
    main()
