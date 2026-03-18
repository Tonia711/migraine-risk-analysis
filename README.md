# Migraine Risk Analysis: End-to-End Data Analytics Pipeline

## 📌 Overview

A production-grade **end-to-end data analytics pipeline** that identifies key risk factors for migraine using large-scale healthcare data (NHIS 2018, 25,000+ records). This project demonstrates full data science workflow capabilities: data engineering, exploratory analysis, feature engineering, predictive modeling, and actionable insight generation.

**Real-world impact:** Generated predictive models (AUC ≈ 0.81) identifying high-risk populations for targeted healthcare interventions, with interpretable features (mental health, sleep quality) that support clinical decision-making.

---

## Key Achievements

- **Predictive Accuracy:** Built 3 machine learning models (Logistic Regression, Decision Tree, Random Forest) with best-in-class AUC of **0.81**
- **Production Code:** Modular, maintainable Python architecture with clean separation of concerns (data → features → models → evaluation)
- **Feature Engineering:** Engineered 10+ domain-specific features from raw 700+ healthcare variables, reducing dimensionality while improving interpretability
- **Real-world Impact:** Identified key modifiable risk factors (sleep quality, mental health) for healthcare intervention strategies
- **Data Handling:** Successfully managed 25,000+ records, 143MB dataset with custom cleaning rules, imputation strategies, and feature transformations

---

## 🎯 Business Value

- **Healthcare Strategy:** Enables targeted screening and prevention programs for high-risk populations
- **Clinical Decision Support:** Provides interpretable models (odds ratios, decision rules) actionable for clinicians
- **Cost Reduction:** Predictive triage can optimize resource allocation for preventive care
- **Public Health:** Supports data-driven insights alignment with SDG 3 (health and wellbeing)

---

## 📊 Dataset

- **Source:** National Health Interview Survey (NHIS) 2018
- **Size:** 25,000+ records, 700+ variables
- **Type:** Cross-sectional health survey data

> ⚠️ Note: Raw dataset is not included in this repository due to size and licensing constraints.
> For public demos, we provide a deterministic modeling-table sample: `Data/sample/final_modeling_table_sample.csv`.
> You can run training and evaluation with `python src/cli.py --data-mode sample --stage all`.
> If you have the raw NHIS CSV, use `--data-mode full` to run the complete pipeline.

---

## 🧠 Analytical Approach

### 1. Data Understanding

- Explored demographic, behavioral, and health-related variables
- Identified target variable (migraine occurrence)

### 2. Data Cleaning

- Handled missing values and invalid codes
- Removed irrelevant and high-missing features

### 3. Feature Engineering

- Constructed domain-specific variables:
  - Mental Distress Index
  - Sleep Sufficiency
  - BMI Categories
  - Age Groups
  - Pain Index

### 4. Modeling

Implemented multiple models to balance interpretability and performance:

- Logistic Regression (primary interpretable model)
- Decision Tree (rule-based insights)
- Random Forest (ensemble model for robustness)

### 5. Evaluation
- Metrics: AUC, Accuracy, Precision, Recall, F1-score
- Threshold tuning: The decision threshold is selected on the validation set (val) using the PR curve to maximize F1.
  Final metrics are computed on the held-out test set (test) to avoid test-set leakage.

---

## Technical Skills Demonstrated

| Category                       | Technologies                           | Application                                                     |
| ------------------------------ | -------------------------------------- | --------------------------------------------------------------- |
| **Data Engineering**           | Pandas, NumPy                          | Data cleaning, imputation, transformation                       |
| **Feature Engineering**        | Domain expertise, Statistical analysis | 10+ engineered features from 700+ raw variables                 |
| **Machine Learning**           | scikit-learn, PySpark MLlib            | Classification (3 models), model evaluation                     |
| **Evaluation & Visualization** | Matplotlib, sklearn.metrics            | ROC, PR curves, confusion matrices, threshold optimization      |
| **Code Architecture**          | Python OOP, modular design             | 4 production-grade modules with clean imports                   |
| **Data Science Workflow**      | KDD methodology                        | Full pipeline: understand → clean → engineer → model → evaluate |

---

## 🚀 How to Run

### Python Environment

Use the project-local `.venv` for this repository. Do not mix it with `conda base`, or package availability may differ between interpreters.

```bash
# Create the virtual environment once
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install project dependencies
python -m pip install -r requirements.txt
```

### Quick Start (Jupyter Notebook)

```bash
# Run the complete analysis
jupyter notebook notebooks/analysis.ipynb
```

**Output:**

- Pre-computed metrics summary: `outputs/dm_metrics_summary.csv`
- Visualizations: 9 publication-quality figures in `outputs/figs/`

### Run Individual Pipeline Stages

```bash
# Public demo (no raw data required)
python src/cli.py --data-mode sample --stage all --output-dir outputs

# Full pipeline (requires your NHIS CSV under Data/raw/)
python src/cli.py --data-mode full --stage all --input-raw Data/raw/samadult.csv --output-dir outputs

# Or run Spark implementation for distributed processing
python spark/spark_pipeline.py --input Data/processed/final_modeling_table.csv
```

---

## 📁 Project Structure

```
migraine-risk-analysis/
├── notebooks/
│   └── analysis.ipynb              # Complete interactive analysis
│
├── src/                            # Production-grade modules
│   ├── data/
│   │   └── data_cleaning.py        # 40+ custom cleaning rules
│   ├── features/
│   │   └── feature_engineering.py  # 10+ feature constructions
│   ├── models/
│   │   ├── modeling.py            # 3 model training pipelines
│   │   └── evaluation.py          # Metrics & visualizations
│   ├── pipeline.py                # Orchestration layer
│   └── cli.py                      # One-line runner for public demo
│
├── spark/
│   └── spark_pipeline.py          # Distributed Spark ML implementation
│
├── Data/
│   ├── raw/                       # Original NHIS data
│   └── processed/                 # Cleaned & engineered datasets
│   └── sample/                    # Fixed-size sample for public demo
│
├── outputs/
│   ├── dm_metrics_summary.csv     # Model comparison metrics
│   └── figs/                      # ROC, PR, confusion matrices
│
└── README.md
```

## 📊 Key Results

### 🔑 Top Risk Factors

- Mental distress (strongest predictor)
- Neck and facial pain
- Sleep insufficiency
- Female gender
- Middle age group

### 📊 Model Performance

- Logistic Regression: AUC ≈ 0.80
- Random Forest: AUC ≈ 0.79
- Decision Tree: AUC ≈ 0.76

---

## 💡 Insights & Business Value

This project demonstrates how data analytics can generate actionable insights in healthcare:

- **Preventive Healthcare:** Identifies modifiable lifestyle factors (sleep, stress) for early intervention
- **Risk Segmentation:** Enables targeted screening for high-risk populations

- **Decision Support:** Provides interpretable models (odds ratios, decision rules) for stakeholders

---

## ⚙️ Tech Stack & Skills

- **Language:** Python 3.8+
- **Data Engineering:** Pandas, NumPy (data cleaning, imputation, transformation)
- **Machine Learning:** scikit-learn (LogisticRegression, DecisionTree, RandomForest)
- **Big Data:** PySpark MLlib (distributed classification, feature transformation)
- **Visualization:** Matplotlib, Seaborn (ROC curves, PR curves, confusion matrices)
- **Methodology:** KDD (Knowledge Discovery in Databases)
- **Best Practices:** Modular architecture, stratified validation, threshold optimization

---

---

## 📌 Future Enhancements (Nice-to-Have)

- Interactive dashboard (Streamlit/Tableau) for clinician use
- Longitudinal analysis tracking risk over time
- Advanced models (XGBoost, hyperparameter tuning)
- Causal inference analysis (identifying truly modifiable factors)
- A/B testing framework for intervention effectiveness

---

## 📌 Project Outputs

The main artifacts generated by the pipeline are:
- `outputs/dm_metrics_summary.csv` (model metrics summary, including `threshold_source = val_best_f1`)
- `outputs/figs/` (ROC/PR curves, confusion matrices, and interpretability plots)
