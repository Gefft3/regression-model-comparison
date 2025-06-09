# Regression Model Comparison Pipeline

This repository presents a complete and well-documented machine learning pipeline for comparing regression models using the California Housing dataset.

## Objective

Predict the median house value based on socioeconomic and geographic features using the following models:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* CatBoost Regressor

## Pipeline Overview

1. **Problem Definition**
   Regression task: predict `MedHouseVal`.

2. **Data Collection**
   Dataset loaded via `sklearn.datasets.fetch_california_housing`.

3. **Exploratory Data Analysis (EDA)**

   * Descriptive statistics
   * Histograms
   * Correlation matrix

4. **Preprocessing**

   * Missing value imputation (median)
   * Feature scaling (standardization)

5. **Modeling**

   * Models integrated in `Pipeline`
   * Evaluation via 5-fold cross-validation
   * Metric: Root Mean Squared Error (RMSE)

6. **Model Evaluation**

   * Best model selected by CV RMSE
   * Final evaluation on hold-out test set

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/regression-model-comparison.git
   cd regression-model-comparison
   ```
