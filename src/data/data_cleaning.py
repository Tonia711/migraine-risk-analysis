from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


BAD_CAT = {7, 8, 9}
BAD_CAT_EXT = BAD_CAT | {97, 98, 99}
BMI_MISS = {9999}

EXPLICIT_RULES: dict[str, dict[str, Any]] = {
    "AMIGR": {"missing_codes": BAD_CAT},
    "HYPEV": {"missing_codes": BAD_CAT},
    "CHDEV": {"missing_codes": BAD_CAT},
    "MIEV": {"missing_codes": BAD_CAT},
    "HRTEV": {"missing_codes": BAD_CAT},
    "CANEV": {"missing_codes": BAD_CAT},
    "AHAYFYR": {"missing_codes": BAD_CAT},
    "AHCSYR1": {"missing_codes": BAD_CAT},
    "PAINLB": {"missing_codes": BAD_CAT},
    "PAINFACE": {"missing_codes": BAD_CAT},
    "SINYR": {"missing_codes": BAD_CAT},
    "ANX_2": {"missing_codes": BAD_CAT},
    "DEP_2": {"missing_codes": BAD_CAT},
    "ARX12_2": {"missing_codes": BAD_CAT},
    "ARX12_3": {"missing_codes": BAD_CAT},
    "ASIHIVT": {"missing_codes": BAD_CAT},
    "ASPONOWN": {"missing_codes": BAD_CAT},
    "DIBPRE2": {"missing_codes": BAD_CAT},
    "JNTSYMP": {"missing_codes": BAD_CAT},
    "PAINECK": {"missing_codes": BAD_CAT},
    "DEP_1": {"missing_codes": BAD_CAT, "valid_range": (1, 5)},
    "ANX_1": {"missing_codes": BAD_CAT, "valid_range": (1, 5)},
    "ASIEFFRT": {"missing_codes": BAD_CAT, "valid_range": (1, 5)},
    "ASIRSTLS": {"missing_codes": BAD_CAT, "valid_range": (1, 5)},
    "ASISAD": {"missing_codes": BAD_CAT, "valid_range": (1, 5)},
    "ASIHOPLS": {"missing_codes": BAD_CAT, "valid_range": (1, 5)},
    "ASINERVE": {"missing_codes": BAD_CAT, "valid_range": (1, 5)},
    "ASINERV": {"missing_codes": BAD_CAT, "valid_range": (1, 5)},
    "ASIWTHLS": {"missing_codes": BAD_CAT, "valid_range": (1, 5)},
    "TIRED_1": {"missing_codes": BAD_CAT, "valid_range": (1, 4)},
    "ASIMEDC": {"missing_codes": BAD_CAT, "valid_range": (1, 4)},
    "ASIRETR": {"missing_codes": BAD_CAT, "valid_range": (1, 4)},
    "ASINBILL": {"missing_codes": BAD_CAT, "valid_range": (1, 4)},
    "ASIMUCH": {"missing_codes": BAD_CAT, "valid_range": (1, 4)},
    "TIRED_2": {"missing_codes": BAD_CAT, "valid_range": (1, 3)},
    "SMKNOW": {"missing_codes": BAD_CAT, "valid_range": (1, 3)},
    "AWORPAY": {"missing_codes": BAD_CAT, "valid_range": (1, 3)},
    "TIRED_3": {"missing_codes": BAD_CAT, "valid_range": (1, 3)},
    "PAIN_2A": {"missing_codes": BAD_CAT, "valid_range": (1, 4)},
    "PAIN_4": {"missing_codes": BAD_CAT, "valid_range": (1, 3)},
    "ANX_3R": {"missing_codes": BAD_CAT, "valid_range": (1, 3)},
    "SMKSTAT2": {"missing_codes": BAD_CAT | {9}},
    "ALCSTAT": {"missing_codes": BAD_CAT | {10}},
    "ALC12MTP": {"missing_codes": BAD_CAT},
    "DOINGLWA": {"missing_codes": BAD_CAT},
    "R_MARITL": {"missing_codes": BAD_CAT},
    "HISPAN_I": {"missing_codes": BAD_CAT},
    "RACERPI2": {"missing_codes": BAD_CAT},
    "AGE_P": {"clip_range": (18, 85)},
    "BMI": {"missing_codes": BMI_MISS, "scale": 0.01, "plausible_range": (10, 80)},
    "ASISLEEP": {"missing_codes": {97, 98, 99}, "clip_range": (1, 24)},
    "YRSWRKPA": {"missing_codes": {97, 98, 99}, "clip_range": (0, 35)},
    "ALC12MYR": {"missing_codes": {997, 998, 999}, "clip_range": (0, 366)},
    "BEDDAYR": {"missing_codes": {997, 998, 999}, "clip_range": (0, 366)},
    "WKDAYR": {"missing_codes": {997, 998, 999}, "clip_range": (0, 366)},
}


def _apply_cleaning_rule(series: pd.Series, rule: dict[str, Any]) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce") if pd.api.types.is_numeric_dtype(series) else series.copy()

    if "missing_codes" in rule:
        cleaned = cleaned.where(~cleaned.isin(rule["missing_codes"]), np.nan)

    if "scale" in rule:
        cleaned = pd.to_numeric(cleaned, errors="coerce") * float(rule["scale"])

    if "valid_range" in rule:
        lo, hi = rule["valid_range"]
        cleaned = pd.to_numeric(cleaned, errors="coerce")
        cleaned = cleaned.where(~((cleaned < lo) | (cleaned > hi)), np.nan)

    if "clip_range" in rule:
        lo, hi = rule["clip_range"]
        cleaned = pd.to_numeric(cleaned, errors="coerce")
        cleaned = cleaned.where(~((cleaned < lo) | (cleaned > hi)), np.nan)

    if "plausible_range" in rule:
        lo, hi = rule["plausible_range"]
        cleaned = pd.to_numeric(cleaned, errors="coerce")
        cleaned = cleaned.where(~((cleaned < lo) | (cleaned > hi)), np.nan)

    return cleaned


def _try_cast_nullable_int(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_float_dtype(series):
        return series
    non_na = series.dropna()
    if non_na.empty:
        return series
    if np.all(np.isclose(non_na % 1, 0, atol=1e-9)):
        return series.astype("Int64")
    return series


def clean_dataframe(df: pd.DataFrame, id_col: str = "RID") -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        if col == id_col:
            continue
        rule = EXPLICIT_RULES.get(col)
        if rule is not None:
            cleaned[col] = _apply_cleaning_rule(cleaned[col], rule)
        else:
            unique_vals = pd.Series(cleaned[col].dropna().unique())
            if not unique_vals.empty:
                share_codes = unique_vals.isin(list(BAD_CAT_EXT | {9999})).mean()
                if share_codes > 0.5:
                    cleaned[col] = cleaned[col].where(~cleaned[col].isin(BAD_CAT_EXT), np.nan)
        cleaned[col] = _try_cast_nullable_int(cleaned[col])
    return cleaned


def basic_imputation(
    df: pd.DataFrame,
    target_col: str = "AMIGR",
    id_col: str = "RID",
    known_continuous: set[str] | None = None,
) -> pd.DataFrame:
    known_continuous = known_continuous or {"AGE_P", "BMI", "ASISLEEP", "WKDAYR", "MODMIN"}
    out = df.copy()

    if target_col in out.columns:
        out = out[out[target_col].notna()].copy()

    candidate_cols = [c for c in out.columns if c not in {id_col, target_col}]
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []

    for col in candidate_cols:
        s = out[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            categorical_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(s):
            nunique = s.dropna().nunique()
            int_like = np.all(np.isclose(s.dropna() % 1, 0, atol=1e-9)) if s.dropna().size else False
            if col not in known_continuous and nunique <= 15 and int_like:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)

    if numeric_cols:
        num_imp = SimpleImputer(strategy="median")
        out[numeric_cols] = num_imp.fit_transform(out[numeric_cols])

    if categorical_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        out[categorical_cols] = cat_imp.fit_transform(out[categorical_cols])

    for col in out.columns:
        if col in {id_col, target_col}:
            continue
        out[col] = _try_cast_nullable_int(out[col])

    return out


def run_data_cleaning(
    input_csv: str | Path,
    cleaned_output_csv: str | Path | None = None,
    imputed_output_csv: str | Path | None = None,
    id_col: str = "RID",
    target_col: str = "AMIGR",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv)

    df_clean = clean_dataframe(df=df, id_col=id_col)
    df_imputed = basic_imputation(df=df_clean, target_col=target_col, id_col=id_col)

    if cleaned_output_csv is not None:
        Path(cleaned_output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(cleaned_output_csv, index=False)

    if imputed_output_csv is not None:
        Path(imputed_output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_imputed.to_csv(imputed_output_csv, index=False)

    return df_clean, df_imputed
