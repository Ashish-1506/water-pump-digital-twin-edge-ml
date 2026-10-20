"""
combine_datasets.py
===================
Combines separate recording session CSV files into a unified training dataset.

Best Practices Implemented:
1. Concatenates individual session CSVs (`normal_session1.csv`, `dryrun_session1.csv`, etc.).
2. Checks and reports label distribution (`value_counts`).
3. Highlights class imbalances and suggests class weights if needed.
4. Preserves only one header line in the output file.
5. Works cleanly with standard library (csv/collections) and leverages pandas if installed.
"""

from pathlib import Path
from collections import Counter
import argparse
import csv

LABEL_NAMES = {
    0: "Normal",
    1: "Dry Run",
    2: "Overload",
    3: "Bearing / Vibration Fault",
    4: "Overheating",
}

EXPECTED_COLS = ["timestamp", "vib_rms", "vib_peak", "vib_x", "vib_y", "vib_z", "current", "temperature", "label"]


def combine_sessions(raw_dir: Path, output_file: Path) -> None:
    csv_files = sorted(raw_dir.glob("*.csv"))
    # Exclude combined or output files
    csv_files = [f for f in csv_files if not f.name.startswith("combined")]

    if not csv_files:
        print(f"[!] No CSV session files found in {raw_dir}")
        return

    print(f"[*] Found {len(csv_files)} session CSV files in {raw_dir}:")
    all_rows = []
    total_files_loaded = 0

    for f in csv_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                if not reader.fieldnames:
                    continue
                missing = set(EXPECTED_COLS) - set(reader.fieldnames)
                if missing:
                    print(f"  [-] Skipping {f.name}: missing columns {missing}")
                    continue

                count = 0
                for row in reader:
                    all_rows.append(row)
                    count += 1
                total_files_loaded += 1
                print(f"  [+] Loaded {f.name} ({count} rows)")
        except Exception as e:
            print(f"  [-] Error reading {f.name}: {e}")

    if not all_rows:
        print("[!] No valid rows loaded to combine.")
        return

    # Count labels
    label_counts = Counter()
    for row in all_rows:
        try:
            lbl = int(row["label"])
            label_counts[lbl] += 1
        except (ValueError, KeyError):
            pass

    total = len(all_rows)
    print("\n" + "=" * 55)
    print("DATASET LABEL DISTRIBUTION (value_counts)")
    print("=" * 55)
    for label_id in sorted(label_counts.keys()):
        count = label_counts[label_id]
        name = LABEL_NAMES.get(label_id, f"Class {label_id}")
        pct = (count / total) * 100
        print(f"  Label {label_id} ({name:<25}): {count:5d} rows ({pct:5.1f}%)")

    # Check for imbalance
    if len(label_counts) > 1:
        counts_list = list(label_counts.values())
        max_c = max(counts_list)
        min_c = min(counts_list)
        ratio = max_c / max(min_c, 1)
        if ratio > 2.0:
            print(f"\n[!] Imbalance detected: Majority class is {ratio:.1f}x the minority class.")
            print("    Recommendation: Record additional sessions for minority classes or use class weights.")

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=EXPECTED_COLS)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({col: r.get(col, "") for col in EXPECTED_COLS})

    print(f"\n[OK] Combined dataset saved to: {output_file} ({total} total rows, 1 header line)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine session CSVs and verify label distribution")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "raw"),
        help="Path to folder containing session CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "processed" / "combined_dataset.csv"),
        help="Path for unified CSV output",
    )
    args = parser.parse_args()

    combine_sessions(Path(args.raw_dir), Path(args.output))
