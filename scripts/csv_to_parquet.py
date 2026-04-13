"""
Convert project CSV files to Parquet for 5-10x faster dashboard load times.

Usage (from project root):
    py scripts/csv_to_parquet.py

The script converts:
    processed_datasets/ueba_dataset_3b.csv           ->  .parquet
    explainability/alert_table/alert_table_3.csv     ->  .parquet

The original CSV files are kept unchanged.
"""

import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_FILES = [
    os.path.join(BASE_DIR, "processed_datasets", "ueba_dataset_4", "ueba_dataset_4a.csv"),
    os.path.join(BASE_DIR, "processed_datasets", "ueba_dataset_4", "ueba_dataset_4b.csv"),
    os.path.join(BASE_DIR, "processed_datasets", "ueba_dataset_4", "ueba_dataset_4_train.csv"),
    os.path.join(BASE_DIR, "processed_datasets", "ueba_dataset_4", "ueba_dataset_4_test_stream.csv"),    
    os.path.join(BASE_DIR, "explainability", "alert_table", "alert_table_4.csv"),
]


def convert(csv_path: str) -> None:
    """
    Convert a single CSV file to Parquet, skipping if already up-to-date.

    Writes the Parquet file alongside the source CSV. Skips conversion if the
    Parquet file exists and is newer than the CSV (mtime comparison). Prints
    status and file-size comparison on conversion.

    Args:
        csv_path: Absolute path to the source CSV file.
        
    Returns:
        None:
    """
    parquet_path = csv_path.rsplit(".", 1)[0] + ".parquet"
    rel = os.path.relpath(csv_path, BASE_DIR)

    if not os.path.exists(csv_path):
        print(f"  SKIP  {rel}  (file not found)")
        return

    if os.path.exists(parquet_path):
        csv_mtime = os.path.getmtime(csv_path)
        pq_mtime = os.path.getmtime(parquet_path)
        if pq_mtime >= csv_mtime:
            print(f"  UP-TO-DATE  {rel}")
            return

    print(f"  CONVERTING  {rel} ... ", end="", flush=True)
    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    csv_size = os.path.getsize(csv_path) / (1024 * 1024)
    pq_size = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"done  ({csv_size:.1f} MB CSV -> {pq_size:.1f} MB Parquet)")


def main() -> None:
    """
    Convert all files listed in CSV_FILES.
    """
    print("CSV -> Parquet converter\n")
    for path in CSV_FILES:
        convert(path)
    print("\nAll done. The dashboard will now load from Parquet automatically.")


if __name__ == "__main__":
    main()
