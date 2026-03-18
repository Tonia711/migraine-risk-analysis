from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def reverse_minmax(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.dropna().values
    if values.size == 0:
        return numeric
    lo, hi = np.nanmin(values), np.nanmax(values)
    return (lo + hi) - numeric


def construct_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "AGE_P" in out.columns:
        out["AGE_BAND"] = pd.cut(
            pd.to_numeric(out["AGE_P"], errors="coerce"),
            bins=[17, 29, 44, 59, np.inf],
            labels=["18-29", "30-44", "45-59", "60+"],
        )

    if "BMI" in out.columns:
        out["BMI_CAT"] = pd.cut(
            pd.to_numeric(out["BMI"], errors="coerce"),
            bins=[0, 18.5, 25, 30, np.inf],
            labels=["Underweight", "Normal", "Overweight", "Obese"],
        )

    if {"ALC12MYR", "ALC12MTP"}.issubset(out.columns):
        days = pd.to_numeric(out["ALC12MYR"], errors="coerce")
        units = pd.to_numeric(out["ALC12MTP"], errors="coerce").replace(0, np.nan)
        out["ALC_INTENSITY"] = (days / units).fillna(0)

    if "YRSWRKPA" in out.columns:
        out["WORK_EXP_CAT"] = pd.cut(
            pd.to_numeric(out["YRSWRKPA"], errors="coerce"),
            bins=[-1, 5, 15, 25, np.inf],
            labels=["0-5", "6-15", "16-25", "26+"],
        )

    pain_freq_map = {1: 0, 2: 1, 3: 2, 4: 3}
    pain_int_map = {1: 1, 3: 2, 2: 3}
    if {"PAIN_2A", "PAIN_4"}.issubset(out.columns):
        pf = pd.to_numeric(out["PAIN_2A"], errors="coerce").map(pain_freq_map)
        pi = pd.to_numeric(out["PAIN_4"], errors="coerce").map(pain_int_map)
        out["PAIN_INDEX"] = (pf * pi).astype("float")

    mental_items_all = ["ASISAD", "ASINERV", "ASIRSTLS", "ASIHOPLS", "ASIEFFRT", "ASIWTHLS"]
    mental_items = [c for c in mental_items_all if c in out.columns]
    if mental_items:
        reversed_items = pd.DataFrame({c: reverse_minmax(out[c]) for c in mental_items})
        out["MENTAL_HEALTH_SCORE"] = reversed_items.sum(axis=1, min_count=1)

    if "ASISLEEP" in out.columns:
        sleep_hours = pd.to_numeric(out["ASISLEEP"], errors="coerce")
        out["SLEEP_SUFFICIENT"] = pd.Series(
            np.where(sleep_hours.between(7, 9, inclusive="both"), 1, 0),
            index=out.index,
        ).astype("Int64")
        out["SLEEP_SUFF_LABEL"] = out["SLEEP_SUFFICIENT"].map({1: "Sufficient", 0: "Insufficient"}).astype("string")

    if {"ASISLEEP", "DOINGLWA"}.issubset(out.columns):
        out["SLEEP_X_WORK"] = pd.to_numeric(out["ASISLEEP"], errors="coerce") * pd.to_numeric(
            out["DOINGLWA"], errors="coerce"
        )

    if {"SEX", "ALCSTAT"}.issubset(out.columns):
        out["SEX_X_ALC"] = pd.to_numeric(out["SEX"], errors="coerce") * pd.to_numeric(out["ALCSTAT"], errors="coerce")

    return out


def run_feature_engineering(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    summary_csv: str | Path | None = None,
) -> pd.DataFrame:
    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv)
    transformed = construct_features(df)

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        transformed.to_csv(output_csv, index=False)

    if summary_csv is not None:
        Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(
            {
                "Remaining NaNs": transformed.isna().sum(),
                "Unique Values": transformed.nunique(),
                "Data Type": transformed.dtypes,
            }
        )
        summary.to_csv(summary_csv)

    return transformed
