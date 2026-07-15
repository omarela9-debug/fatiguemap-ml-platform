import argparse
import csv
import json
import re
from pathlib import Path


RUNS_DIR = Path(__file__).resolve().parent / "runs"
RUN_FILE_RE = re.compile(r"run_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.csv$")

TRUE_RAW_COLUMNS = {
    "heart_rate_raw_line",
}

TRUE_RAW_PREFIXES = (
    "raw_imu_",
    "raw_accel_",
    "raw_gyro_",
    "raw_mag_",
    "raw_eye_timestamp",
    "raw_eye_camera",
    "raw_eye_width",
    "raw_eye_height",
    "raw_eye_channels",
    "raw_eye_dtype",
    "raw_eye_mean",
    "raw_eye_std",
    "raw_eye_frame_count",
)

RAW_LIKE_TOKENS = (
    "raw",
    "accel",
    "gyro",
    "mag",
    "imu",
    "eye",
    "pupil",
    "iris",
    "closure",
    "perclos",
    "blink",
    "heart_rate",
)


def latest_run():
    files = sorted(
        (path for path in RUNS_DIR.glob("*.csv") if RUN_FILE_RE.fullmatch(path.name)),
        key=lambda path: path.stat().st_mtime,
    )

    if not files:
        raise FileNotFoundError(f"No CSV files found in {RUNS_DIR}")

    return files[-1]


def matching_columns(columns, true_raw_only=False):
    if true_raw_only:
        return [
            col
            for col in columns
            if col in TRUE_RAW_COLUMNS
            or any(col.startswith(prefix) for prefix in TRUE_RAW_PREFIXES)
        ]

    return [
        col
        for col in columns
        if any(token in col.lower() for token in RAW_LIKE_TOKENS)
    ]


def has_selected_value(row, columns):
    for col in columns:
        value = str(row.get(col, "")).strip()

        if not value:
            continue

        if col == "raw_eye_frame_count" and value in {"0", "0.0"}:
            continue

        return True

    return False


def read_preview(csv_path, columns, limit, drop_empty=False):
    rows = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if drop_empty and not has_selected_value(row, columns):
                continue

            rows.append({col: row.get(col, "") for col in columns})

            if len(rows) >= limit:
                break

    return rows


def count_rows(csv_path):
    with csv_path.open(newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def export_columns(csv_path, columns, out_path, drop_empty=False):
    with csv_path.open(newline="") as src, out_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=columns)
        writer.writeheader()

        for row in reader:
            if drop_empty and not has_selected_value(row, columns):
                continue

            writer.writerow({col: row.get(col, "") for col in columns})


def main():
    parser = argparse.ArgumentParser(
        description="Inspect or export raw/sensor-like columns from a FatigueMap run CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        help="Path to a run CSV. Defaults to the newest backend/runs CSV.",
    )
    parser.add_argument(
        "--true-raw-only",
        action="store_true",
        help="Only select columns that start with raw_ or store raw serial lines.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of selected rows to print as JSON.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Write selected columns to this CSV path.",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Skip rows where every selected column is blank.",
    )
    args = parser.parse_args()

    csv_path = args.csv_path or latest_run()
    csv_path = csv_path.expanduser().resolve()

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []

    selected = matching_columns(columns, true_raw_only=args.true_raw_only)
    true_raw = matching_columns(columns, true_raw_only=True)

    print(f"file: {csv_path}")
    print(f"rows: {count_rows(csv_path)}")
    print(f"columns: {len(columns)}")
    print(f"true_raw_columns: {len(true_raw)}")

    if true_raw:
        print(json.dumps(true_raw, indent=2))
    else:
        print("No true raw columns are present in this CSV.")
        print("This run was recorded before raw_accel/raw_gyro/raw_eye logging was added.")

    print(f"selected_columns: {len(selected)}")
    print(json.dumps(selected, indent=2))

    if args.preview > 0 and selected:
        print("preview:")
        print(json.dumps(read_preview(csv_path, selected, args.preview, args.drop_empty), indent=2))

    if args.export:
        out_path = args.export.expanduser().resolve()
        export_columns(csv_path, selected, out_path, args.drop_empty)
        print(f"exported: {out_path}")


if __name__ == "__main__":
    main()
