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
| Tree-Based | CatBoost, LightGBM |

### Why LightGBM?

- Handles mixed feature types (numeric, categorical, sparse text features)
- No feature scaling required
- Captures non-linear interactions effectively
- Scalable and efficient with high-dimensional TF-IDF vectors

---

## Results

### Best Model Performance (LightGBM with TF-IDF + Numeric Features)

| Metric | Value |
|--------|-------|
| **MAE** | 262 |
| **RMSE** | 351 |
| **R²** | 0.80 |
| **Spearman Correlation** | 0.90 |

### Model Comparison

| Model | Features | R² | Spearman |
|-------|----------|-----|----------|
| Linear Regression | Numeric Only | — | — |
| Ridge Regression | Numeric Only | — | — |
| LinearSVR | TF-IDF + Numeric | — | — |
| CatBoost | Numeric Only | — | — |
| **LightGBM** | **TF-IDF + Numeric** | **0.80** | **0.90** |

---

## Key Findings

| Finding | Implication |
|---------|-------------|
| Problem index (contest order) is the strongest predictor (ρ ≈ 0.74) | Contest structure encodes difficulty progression |
| Time limit shows moderate correlation (ρ ≈ 0.46) | Harder problems allow more computational time |
| Textual features provide complementary semantic signals | Longer descriptions weakly correlate with higher difficulty |
| Tree-based models outperform linear models | Difficulty prediction requires capturing non-linear patterns |
| Spearman correlation > 0.90 | Model preserves relative difficulty ordering effectively |

---

## MLOps: Experiment Tracking with MLflow

All experiments were tracked using MLflow to ensure reproducibility:

- Hyperparameters and evaluation metrics logged for each run
- Numeric-only and text-augmented experiments compared
- Final model pipeline stored as versioned artifact
- Seamless integration with scikit-learn pipelines

**Advantages:** Centralized dashboard for model comparison, reproducible preprocessing, minimal setup overhead.

---
