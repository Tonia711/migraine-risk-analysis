from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def make_modeling_table_sample(
    input_csv: str | Path = "Data/processed/final_modeling_table.csv",
    output_csv: str | Path = "Data/sample/final_modeling_table_sample.csv",
    n: int = 2000,
    seed: int = 42,
) -> None:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", newline="", encoding="utf-8") as f_in:
        reader = csv.reader(f_in)
        header = next(reader)

        if "AMIGR" not in header:
            raise ValueError("input must contain 'AMIGR' column")

        rid_idx = header.index("RID") if "RID" in header else None
        rng = random.Random(seed)

        # Reservoir sampling: O(N) streaming, O(n) memory.
        # This avoids pandas/numpy so the sample generation works even in constrained environments.
        reservoir: list[list[str]] = []
        for i, row in enumerate(reader):
            if i < n:
                reservoir.append(row)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = row

    if rid_idx is not None:
        def _rid_key(r: list[str]) -> float:
            v = r[rid_idx]
            try:
                return float(v)
            except ValueError:
                return float("inf")

        reservoir.sort(key=_rid_key)

    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        writer.writerows(reservoir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed-size sample for public demo.")
    parser.add_argument("--input", default="Data/processed/final_modeling_table.csv")
    parser.add_argument("--output", default="Data/sample/final_modeling_table_sample.csv")
    parser.add_argument("--n", type=int, default=2000, help="Number of rows to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_modeling_table_sample(
        input_csv=args.input,
        output_csv=args.output,
        n=args.n,
        seed=args.seed,
    )

