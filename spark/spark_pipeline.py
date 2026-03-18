from __future__ import annotations

import argparse
from pathlib import Path


def run_spark_pipeline(
    input_path: str | Path,
    output_dir: str | Path = "outputs/spark",
    target_col: str = "AMIGR",
    id_col: str = "RID",
) -> None:
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spark = SparkSession.builder.appName("MigraineRiskSparkPipeline").getOrCreate()

    input_path = str(input_path)
    if input_path.endswith(".parquet"):
        sdf = spark.read.parquet(input_path)
    else:
        sdf = spark.read.csv(input_path, header=True, inferSchema=True)

    if target_col not in sdf.columns:
        raise ValueError(f"target col '{target_col}' not in input data")

    sdf = sdf.filter(F.col(target_col).isNotNull())
    sdf = sdf.withColumn("label", F.when(F.col(target_col) == 1, F.lit(1.0)).otherwise(F.lit(0.0)))

    feature_cols = [c for c in sdf.columns if c not in {id_col, target_col, "label"}]
    numeric_cols = [c for c, t in sdf.dtypes if c in feature_cols and t in {"int", "bigint", "double", "float", "long"}]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    stages = []
    encoded_cols = []
    for c in categorical_cols:
        idx_col = f"{c}_idx"
        ohe_col = f"{c}_ohe"
        stages.append(StringIndexer(inputCol=c, outputCol=idx_col, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[idx_col], outputCols=[ohe_col], handleInvalid="keep"))
        encoded_cols.append(ohe_col)

    assembler_inputs = numeric_cols + encoded_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")

    train_df, test_df = sdf.randomSplit([0.7, 0.3], seed=42)

    models = {
        "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=200),
        "RandomForest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=200, seed=42),
    }

    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    rows = []

    for name, clf in models.items():
        pipeline = Pipeline(stages=stages + [assembler, clf])
        fitted = pipeline.fit(train_df)
        pred = fitted.transform(test_df)
        auc = evaluator.evaluate(pred)
        rows.append((name, float(auc)))

    result_df = spark.createDataFrame(rows, ["model", "auc"])
    result_df.toPandas().to_csv(output_dir / "spark_metrics_summary.csv", index=False)

    spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Spark-based migraine risk pipeline.")
    parser.add_argument(
        "--input",
        default="Data/processed/final_modeling_table.csv",
        help="Path to the modeling table CSV or Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/spark",
        help="Directory for Spark pipeline outputs.",
    )
    parser.add_argument(
        "--target-col",
        default="AMIGR",
        help="Target column name.",
    )
    parser.add_argument(
        "--id-col",
        default="RID",
        help="Identifier column name to exclude from features.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_spark_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        target_col=args.target_col,
        id_col=args.id_col,
    )
