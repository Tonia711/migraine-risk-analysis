from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def _normalize_target(y: pd.Series) -> pd.Series:
    y_numeric = pd.to_numeric(y, errors="coerce")
    unique = set(y_numeric.dropna().unique().tolist())
    if unique.issubset({1, 2}):
        return (y_numeric == 1).astype(int)
    return y_numeric.astype(int)


def prepare_xy(
    df: pd.DataFrame,
    target_col: str = "AMIGR",
    id_col: str = "RID",
    drop_leakage: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    working = df.copy()

    if drop_leakage:
        leak_cols = [c for c in working.columns if target_col in c and c != target_col]
        if leak_cols:
            working = working.drop(columns=leak_cols)

    y = _normalize_target(working[target_col])
    X = working.drop(columns=[c for c in [id_col, target_col] if c in working.columns])
    X = pd.get_dummies(X, drop_first=False)
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.30,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.30,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split into train/val/test with stratification.
    val_size is expressed as a fraction of the whole dataset (not of the train split).
    """
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1.")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    remaining = 1.0 - test_size
    rel_val_size = val_size / remaining

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=rel_val_size,
        random_state=random_state,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_models(random_state: int = 42) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            solver="saga",
            penalty="l1",
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=2,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, object] | None = None,
) -> dict[str, object]:
    models = models or build_models()
    fitted: dict[str, object] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def run_modeling(
    input_csv: str | Path,
    target_col: str = "AMIGR",
    id_col: str = "RID",
    test_size: float = 0.30,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, object]:
    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv)

    X, y = prepare_xy(df, target_col=target_col, id_col=id_col)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X,
        y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    fitted_models = train_models(X_train=X_train, y_train=y_train)

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "models": fitted_models,
    }
