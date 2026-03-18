from __future__ import annotations

from pathlib import Path

from data.data_cleaning import run_data_cleaning
from features.feature_engineering import run_feature_engineering
from models.evaluation import evaluate_models
from models.modeling import run_modeling


def run_pipeline(
    raw_input_csv: str | Path = "Data/raw/samadult.csv",
    processed_dir: str | Path = "Data/processed",
    output_dir: str | Path = "outputs",
) -> None:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intermediate outputs in Data/processed
    cleaned_csv = processed_dir / "final_selected_table_clean.csv"
    imputed_csv = processed_dir / "final_selected_table_imputed.csv"
    modeled_csv = processed_dir / "final_modeling_table.csv"
    construction_summary_csv = processed_dir / "construction_summary_table.csv"

    _, _ = run_data_cleaning(
        input_csv=raw_input_csv,
        cleaned_output_csv=cleaned_csv,
        imputed_output_csv=imputed_csv,
    )

    _ = run_feature_engineering(
        input_csv=imputed_csv,
        output_csv=modeled_csv,
        summary_csv=construction_summary_csv,
    )

    modeling_result = run_modeling(input_csv=modeled_csv)
    metrics_df = evaluate_models(
        models=modeling_result["models"],
        X_test=modeling_result["X_test"],
        y_test=modeling_result["y_test"],
        save_dir=output_dir / "figs",
    )
    metrics_df.to_csv(output_dir / "dm_metrics_summary.csv", index=False)


if __name__ == "__main__":
    run_pipeline()
