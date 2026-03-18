# Migraine Risk Analysis: Data-Driven Insights from Large-Scale Healthcare Data

## 📌 Overview

This project presents an end-to-end data analytics and machine learning pipeline applied to a large-scale healthcare dataset (NHIS 2018). The objective is to identify key risk factors associated with migraine and generate actionable insights to support data-driven decision-making in public health and healthcare systems.

Unlike purely academic analysis, this project emphasizes **real-world applicability**, combining statistical analysis, machine learning, and interpretable outputs to bridge the gap between data and decision-making.

---

## 🎯 Objectives

* Identify high-risk demographic groups for migraine
* Analyze modifiable risk factors (e.g., sleep, mental health, BMI, lifestyle)
* Build interpretable predictive models for risk classification
* Translate analytical results into actionable healthcare insights

---

## 📊 Dataset

* **Source:** National Health Interview Survey (NHIS) 2018
* **Size:** 25,000+ records, 700+ variables
* **Type:** Cross-sectional health survey data

> ⚠️ Note: Raw dataset is not included in this repository due to size and licensing constraints.

---

## 🧠 Analytical Approach

### 1. Data Understanding

* Explored demographic, behavioral, and health-related variables
* Identified target variable (migraine occurrence)

### 2. Data Cleaning

* Handled missing values and invalid codes
* Removed irrelevant and high-missing features

### 3. Feature Engineering

* Constructed domain-specific variables:

  * Mental Distress Index
  * Sleep Sufficiency
  * BMI Categories
  * Age Groups
  * Pain Index

### 4. Modeling

Implemented multiple models to balance interpretability and performance:

* Logistic Regression (primary interpretable model)
* Decision Tree (rule-based insights)
* Random Forest (ensemble model for robustness)

### 5. Evaluation

* Metrics: AUC, Accuracy, Precision, Recall, F1-score
* Best performance:

  * **AUC ≈ 0.80**
* Used stratified train-test split and cross-validation

---

## 📈 Key Results

### 🔑 Top Risk Factors

* Mental distress (strongest predictor)
* Neck and facial pain
* Sleep insufficiency
* Female gender
* Middle age group

### 📊 Model Performance

* Logistic Regression: AUC ≈ 0.80
* Random Forest: AUC ≈ 0.79
* Decision Tree: AUC ≈ 0.76

---

## 💡 Insights & Business Value

This project demonstrates how data analytics can generate actionable insights in healthcare:

* **Preventive Healthcare:** Identifies modifiable lifestyle factors (sleep, stress) for early intervention
* **Risk Segmentation:** Enables targeted screening for high-risk populations
* **Decision Support:** Provides interpretable models (odds ratios, decision rules) for stakeholders
* **Policy Implications:** Supports data-driven public health strategies aligned with SDG 3

---

## ⚙️ Tech Stack

* **Programming:** Python (pandas, NumPy, scikit-learn)
* **Big Data:** PySpark (MLlib)
* **Analytics Tools:** IBM SPSS Modeler
* **Visualization:** Matplotlib
* **Methodology:** KDD (Knowledge Discovery in Databases)

---

## 📁 Project Structure

```
migraine-risk-analysis/
│
├── data/
│   ├── raw/              # Not included
│   ├── processed/
│
├── notebooks/
│   ├── analysis.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── modeling.py
│
├── outputs/
│   ├── figures/
│   ├── results/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook
```

---

## 📌 Future Improvements

* Deploy as an interactive dashboard (e.g., Streamlit / Tableau)
* Integrate real-time or longitudinal datasets
* Apply advanced models (e.g., XGBoost, deep learning)
* Enhance causal inference analysis

---

## 🤝 About This Project

This project was developed as part of a data mining and analytics study, with a focus on applying data science techniques to real-world healthcare challenges.

It reflects an end-to-end workflow from raw data to actionable insights, demonstrating skills in:

* Data preprocessing
* Feature engineering
* Machine learning modeling
* Insight generation and communication

---

## 📬 Contact

If you have any questions or would like to collaborate, feel free to connect via LinkedIn.