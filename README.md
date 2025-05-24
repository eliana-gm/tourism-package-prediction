# Tourism Package Purchase Prediction

This project builds a supervised learning pipeline to predict whether a customer is likely to purchase a newly introduced travel package. The model is designed to assist the "Visit With Us" tourism company in strategically targeting high-potential customers based on demographic and behavioral features. The data was provided by MIT IDSS within my enrollment in the course "DATA SCIENCE AND MACHINE LEARNING: MAKING DATA-DRIVEN DECISIONS."

---

## Project Structure

tourism-purchase-prediction/
|-- data/
│  |-- README.md # Context for absent data
|-- notebooks/
│-- tourism-purchase-prediction.ipynb # Full EDA and model walkthrough
| -- docs/
| --- index.html # GitHub pages design
|-- scripts/
| --- model_pipeline.py # Clean modeling pipeline
│ --- utils.py # Helper plotting functions
|-- report/
| --- summary.md # EDA + insights + modeling reflection
|-- results/
| --- final_metrics.txt # Final model evaluation report
|-- images/
| --- age_distribution.png # Distributions, barplots, and PR curves
| --- income_distribution.png
| --- trips_distribution.png
| --- marital_status_vs_target.png
│ --- productpitched_vs_target.png
│ --- passport_vs_target.png
│ --- designation_vs_target.png
│ --- feature_importance_dt.png
│ --- feature_importance_rf.png
│ --- pr_curve_logreg.png
│ --- pr_curve_svm_rbf.png
│ --- pr_curve_dt.png
|-- README.md # You're here
|__ requirements.txt # Dependencies

---

## Problem Overview

The task is a binary classification problem:

- **Target**: `ProdTaken` (1 = Purchased, 0 = Did not purchase)
- The dataset includes 20+ features: age, income, occupation, passport status, marital status, pitch satisfaction, etc.

---

## Exploratory Analysis & Business Insights

The EDA revealed critical patterns:

- **Single customers** are more likely to convert than married ones.
- **Executives** convert more frequently than upper-level designations.
- Customers **with a passport** and those who **self-inquire** show higher purchase rates.
- The **basic and standard packages** had higher conversion rates than deluxe options.

See [`report/summary.md`](report/summary.md) for full findings and interpretation.

---

## Modeling Techniques

The following models were developed and evaluated:

- Logistic Regression
- Support Vector Machine (Linear + RBF kernels)
- Decision Tree (baseline + tuned)
- Random Forest (baseline)

Hyperparameter tuning was done via `GridSearchCV`, focusing on maximizing **recall** due to business needs (avoiding loss of potential customers).

---

## Final Model Performance

The final model used an **SVM with RBF kernel**, with an optimal threshold determined via PR curve.

See [`results/final_metrics.txt`](results/final_metrics.txt) for full details.

**Best Model: SVM (RBF) with threshold 0.17**
- Recall: 0.69
- Precision: 0.48
- F1 Score: 0.56
- RMSE: 0.89

Visual metrics:
- [`images/pr_curve_svm_rbf.png`](images/pr_curve_svm_rbf.png)
- [`images/feature_importance_rf.png`](images/feature_importance_rf.png)

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Model Pipeline
```bash
python scripts/model_pipeline.py
```

## Resources

Dataset from MIT IDSS: "Data Science and Machine Learning: Making Data-Driven Decisions"

Original assignment: Classification and Hypothesis Testing Practice Project

## Author
Eliana Gabriela Matos Polanco
Email: elianagmpolanco@gmail.com
GitHub: @elianagm

Please see the project report and business conclusions in report/summary.md, or in the GitHub page for [`this project`](https://elianagm.github.io/tourism-package-prediction/)