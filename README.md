# Codeforces Problem Difficulty Prediction

A machine learning model to predict the difficulty rating of competitive programming problems using problem statements and contest metadata.

**Course:** CSCI4734 Machine Learning  
**Institution:** ADA University, School of Information Technologies and Engineering  
**Semester:** Fall 2025  

---

## Project Overview

Competitive programming platforms assign difficulty ratings to problems based on expert judgment and contest outcomes. This manual process is time-consuming, subjective, and difficult to scale. This project builds a supervised regression model to automatically predict problem difficulty from textual and structural features.

**Task Type:** Supervised Regression (continuous numerical output: difficulty rating)

---

## Dataset

Dataset used for this project is available at [Hugging Face](https://huggingface.co/datasets/open-r1/codeforces)

| Attribute | Description |
|-----------|-------------|
| **Size** | 9,556 Codeforces problems |
| **Features** | Problem statements, titles, input/output formats, time/memory limits, contest metadata |
| **Target** | Problem difficulty rating (continuous) |

---

## Methodology

### Feature Engineering

| Feature Type | Features |
|--------------|---------|
| **Structural** | Problem index (encoded), time limit, memory limit |
| **Metadata** | Contest ID, contest type (one-hot encoded) |
| **Textual** | TF-IDF vectors from concatenated problem statements |
| **Temporal** | Contest month, day of week, contest age |

### Models Evaluated

| Category | Models |
|----------|--------|
| Baseline | Linear Regression, Ridge Regression |
| Linear | Support Vector Regression (LinearSVR) |
| Tree-Based | Random Forest, XGBoost, CatBoost, LightGBM |

---

## Results

### Model Performance Comparison

| Model                     | Features Used        | RMSE ↓    | R² ↑     | Spearman ↑ |
| ------------------------- | -------------------- | --------- | -------- | ---------- |
| Linear Regression         | Numeric only         | 605.5     | 0.26     | 0.715      |
| Ridge Regression          | Numeric only         | 605.5     | 0.26     | 0.715      |
| Random Forest             | Numeric only         | 401.1     | 0.68     | 0.827      |
| XGBoost                   | Numeric only         | 359.3     | 0.74     | 0.864      |
| CatBoost                  | Numeric only         | 353.8     | 0.75     | 0.869      |
| LinearSVR                 | TF-IDF + Numeric     | 769.8     | -0.41    | 0.788      |
| Ridge Regression          | TF-IDF + Numeric     | 498.0     | 0.48     | 0.742      |
| LightGBM (tuned)          | TF-IDF + Numeric     | 360.4     | 0.73     | 0.859      |
| **LightGBM (Final Test)** | **TF-IDF + Numeric** | **351.3** | **0.80** | **0.902**  |


### Key Observations

| Finding | Insight |
|---------|---------|
| Linear models underperform | Ridge (numeric only) achieves R² 0.605 vs LightGBM's stronger ranking performance |
| TF-IDF + numeric benefits tree models | Ridge with TF-IDF achieves R² 0.498, lower than numeric-only tree models |
| LinearSVR with text performs poorly | Spearman -0.41 indicates linear models struggle with sparse, high-dimensional text features |
| LightGBM achieves best ranking | Spearman 0.902 on final test shows excellent relative difficulty ordering |
| CatBoost and XGBoost strong on numeric-only | Both achieve R² > 0.86 Spearman without text features |

---

## Key Findings

| Finding | Implication |
|---------|-------------|
| Problem index is the strongest predictor | Contest order encodes difficulty progression |
| Time limit shows moderate correlation | Harder problems allow more computational time |
| Tree-based models outperform linear models | Difficulty prediction requires capturing non-linear patterns |
| Text features benefit ranking when paired with tree models | TF-IDF + LightGBM achieves best Spearman correlation |
| LinearSVR with TF-IDF fails | High-dimensional sparse features degrade linear model performance |

---

## MLOps: Experiment Tracking with MLflow

All experiments were tracked using MLflow to ensure reproducibility:

- Hyperparameters and evaluation metrics logged for each run
- Numeric-only and text-augmented experiments compared
- Final model pipeline stored as versioned artifact

**Advantages:** Centralized dashboard for model comparison, reproducible preprocessing, minimal setup overhead.

