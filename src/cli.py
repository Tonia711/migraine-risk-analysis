from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migraine risk analytics pipeline runner.")
    parser.add_argument(
        "--stage",
        choices=["clean", "features", "model", "all"],
        default="all",
        help="Which pipeline stages to run.",
    )
    parser.add_argument(
        "--data-mode",
        choices=["sample", "full"],
        default="sample",
        help="Use public demo sample data (no raw needed) or full raw-based pipeline.",
    )
    parser.add_argument(
        "--input-raw",
        default="Data/raw/samadult.csv",
        help="Path to raw NHIS CSV (required for data-mode=full).",
    )
    parser.add_argument(
        "--processed-dir",
        default="Data/processed",
        help="Directory for intermediate processed CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write evaluation outputs (metrics + figures).",
    )
    parser.add_argument(
        "--input-modeling-table",
        default="Data/sample/final_modeling_table_sample.csv",
        help="Path to modeling-table CSV (used for data-mode=sample).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure paths exist early for better UX.
    if args.data_mode == "full" and not Path(args.input_raw).exists():
        raise FileNotFoundError(
            f"raw input not found: {args.input_raw}. Provide NHIS CSV under Data/raw/."
        )

    run_pipeline(
        raw_input_csv=args.input_raw,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        stage=args.stage,
        data_mode=args.data_mode,
        input_modeling_table_csv=args.input_modeling_table,
    )


if __name__ == "__main__":
    main()

