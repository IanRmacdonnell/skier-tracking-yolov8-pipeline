import argparse
import csv
import json
from pathlib import Path


FIELDNAMES = ["frame", "x1", "y1", "x2", "y2", "width", "height"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove obviously bad skier keyframe boxes.")
    parser.add_argument("--input", default="annotations/keyframes.csv", help="Input keyframe CSV.")
    parser.add_argument("--output", default="annotations/keyframes_clean.csv", help="Cleaned keyframe CSV.")
    parser.add_argument("--report", default="output/label_audit/clean_report.json", help="JSON report path.")
    parser.add_argument("--min-width", type=float, default=8.0, help="Minimum valid box width in pixels.")
    parser.add_argument("--min-height", type=float, default=8.0, help="Minimum valid box height in pixels.")
    parser.add_argument("--min-area", type=float, default=120.0, help="Minimum valid box area in pixels.")
    return parser.parse_args()


def box_stats(row: dict) -> tuple[float, float, float]:
    x1, y1, x2, y2 = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
    width = x2 - x1
    height = y2 - y1
    return width, height, width * height


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    with input_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    kept = []
    removed = []
    for row in rows:
        width, height, area = box_stats(row)
        reasons = []
        if width < args.min_width:
            reasons.append("min_width")
        if height < args.min_height:
            reasons.append("min_height")
        if area < args.min_area:
            reasons.append("min_area")
        if reasons:
            removed.append({"frame": int(row["frame"]), "width": width, "height": height, "area": area, "reasons": reasons})
        else:
            kept.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in kept:
            writer.writerow(row)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "removed": removed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Kept {len(kept)} of {len(rows)} keyframes")
    print(f"Removed {len(removed)} bad keyframes")
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
