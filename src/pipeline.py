from __future__ import annotations

from pathlib import Path
from typing import Literal

Stage = Literal["clean", "features", "model", "all"]
DataMode = Literal["sample", "full"]


def run_pipeline(
    raw_input_csv: str | Path = "Data/raw/samadult.csv",
    processed_dir: str | Path = "Data/processed",
    output_dir: str | Path = "outputs",
    stage: Stage = "all",
    data_mode: DataMode = "sample",
    input_modeling_table_csv: str | Path | None = None,
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

    if data_mode == "sample":
        # In public demo mode we skip raw cleaning/feature engineering and
        # directly train/evaluate from a small, deterministic modeling-table sample.
        from models.evaluation import evaluate_models
        from models.modeling import run_modeling

        modeling_input = (
            Path(input_modeling_table_csv)
            if input_modeling_table_csv is not None
            else Path("Data/sample/final_modeling_table_sample.csv")
        )
        if stage in {"clean", "features"}:
            raise ValueError("sample mode supports only 'model'/'all' stages.")

        if stage in {"model", "all"}:
            modeling_result = run_modeling(input_csv=modeling_input)
            metrics_df = evaluate_models(
                models=modeling_result["models"],
                X_val=modeling_result["X_val"],
                y_val=modeling_result["y_val"],
                X_test=modeling_result["X_test"],
                y_test=modeling_result["y_test"],
                feature_names=list(modeling_result["X_test"].columns),
                save_dir=output_dir / "figs",
            )
            metrics_df.to_csv(output_dir / "dm_metrics_summary.csv", index=False)
        return

    # full mode
    from data.data_cleaning import run_data_cleaning
    from features.feature_engineering import run_feature_engineering
    from models.evaluation import evaluate_models
    from models.modeling import run_modeling

    if stage in {"clean", "all"}:
        if not Path(raw_input_csv).exists():
            raise FileNotFoundError(
                f"raw input not found: {raw_input_csv}. Provide NHIS CSV under Data/raw/."
            )

        _, _ = run_data_cleaning(
            input_csv=raw_input_csv,
            cleaned_output_csv=cleaned_csv,
            imputed_output_csv=imputed_csv,
        )

    if stage in {"features", "all"}:
        if not imputed_csv.exists():
            raise FileNotFoundError(
                f"imputed CSV missing: {imputed_csv}. Run stage 'clean' first or set stage='all'."
            )

        _ = run_feature_engineering(
            input_csv=imputed_csv,
            output_csv=modeled_csv,
            summary_csv=construction_summary_csv,
        )

    if stage in {"model", "all"}:
        if not modeled_csv.exists():
            raise FileNotFoundError(
                f"modeling-table CSV missing: {modeled_csv}. Run stage 'features' first or set stage='all'."
            )

        modeling_result = run_modeling(input_csv=modeled_csv)
        metrics_df = evaluate_models(
            models=modeling_result["models"],
            X_val=modeling_result["X_val"],
            y_val=modeling_result["y_val"],
            X_test=modeling_result["X_test"],
            y_test=modeling_result["y_test"],
            feature_names=list(modeling_result["X_test"].columns),
            save_dir=output_dir / "figs",
        )
        metrics_df.to_csv(output_dir / "dm_metrics_summary.csv", index=False)


if __name__ == "__main__":
    run_pipeline()
