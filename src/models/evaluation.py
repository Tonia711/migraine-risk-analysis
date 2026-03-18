from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance


def best_threshold_f1(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.r_[0, thresholds]
    f1_values = 2 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.nanargmax(f1_values))
    return float(thresholds[idx]), float(precision[idx]), float(recall[idx]), float(f1_values[idx])


def evaluate_at_threshold(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def plot_roc(y_true: pd.Series, y_proba: np.ndarray, model_name: str, output_path: str | Path | None = None) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {model_name}")
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
    plt.close()
    return float(auc)


def plot_pr(y_true: pd.Series, y_proba: np.ndarray, model_name: str, output_path: str | Path | None = None) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall — {model_name}")
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
    plt.close()
    return float(ap)


def plot_confusion(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str | Path | None = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_models(
    models: dict[str, object],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str] | None = None,
    save_dir: str | Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    base_dir = Path(save_dir) if save_dir is not None else None
    feature_names = feature_names or list(X_test.columns)

    for name, model in models.items():
        y_proba_val = model.predict_proba(X_val)[:, 1]
        threshold, _, _, _ = best_threshold_f1(y_val, y_proba_val)

        y_proba_test = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_at_threshold(y_test, y_proba_test, threshold)
        metrics["model"] = name
        metrics["auc"] = float(roc_auc_score(y_test, y_proba_test))
        metrics["ap"] = float(average_precision_score(y_test, y_proba_test))
        metrics["threshold_source"] = "val_best_f1"
        metrics["precision_at_best_f1"] = metrics["precision"]
        metrics["recall_at_best_f1"] = metrics["recall"]
        metrics["f1_at_best_f1"] = metrics["f1"]
        rows.append(metrics)

        if base_dir is not None:
            safe_name = name.lower().replace(" ", "_")
            plot_roc(y_test, y_proba_test, name, base_dir / f"roc_{safe_name}.png")
            plot_pr(y_test, y_proba_test, name, base_dir / f"pr_{safe_name}.png")
            y_pred = (y_proba_test >= threshold).astype(int)
            plot_confusion(y_test, y_pred, name, base_dir / f"cm_{safe_name}.png")

            # --- Interpretability plots (简历可讲的可视化证据) ---
            # 1) Logistic Regression: odds ratios
            if hasattr(model, "coef_"):
                coefs = np.ravel(model.coef_)
                if coefs.size == len(feature_names):
                    # odds ratio = exp(beta). Rank by absolute log-odds magnitude.
                    log_odds = coefs
                    odds_ratios = np.exp(log_odds)
                    order = np.argsort(-np.abs(log_odds))[:15]
                    top_features = [feature_names[i] for i in order]
                    top_odds = odds_ratios[order]

                    plt.figure(figsize=(8, 5))
                    plt.barh(top_features[::-1], top_odds[::-1])
                    plt.xlabel("Odds Ratio (exp(beta))")
                    plt.title(f"Top Odds Ratios — {name}")
                    plt.tight_layout()
                    plt.savefig(base_dir / "odds_ratio_topk.png", dpi=200)
                    plt.close()

            # 2) Tree models: permutation importance (ROC-AUC)
            if hasattr(model, "predict_proba"):
                try:
                    pi = permutation_importance(
                        model,
                        X_test,
                        y_test,
                        scoring="roc_auc",
                        n_repeats=3,
                        random_state=42,
                        n_jobs=-1,
                    )
                    importances = pi.importances_mean
                    order = np.argsort(-importances)[:20]
                    top_features = [feature_names[i] for i in order]
                    top_importances = importances[order]

                    plt.figure(figsize=(8, 6))
                    plt.barh(top_features[::-1], top_importances[::-1])
                    plt.xlabel("Permutation Importance (mean ROC-AUC decrease)")
                    plt.title(f"Permutation Importance — {name}")
                    plt.tight_layout()
                    plt.savefig(base_dir / f"permutation_importance_{safe_name}.png", dpi=200)
                    plt.close()
                except Exception:
                    # Keep evaluation robust even if permutation importance fails for a specific estimator.
                    pass

    return pd.DataFrame(rows)
