# Regression Model Comparison Pipeline

This repository presents a complete and well-documented machine learning pipeline for comparing regression models using the California Housing dataset.

## Objective

Predict the median house value based on socioeconomic and geographic features using:

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
   - Descriptive statistics  
   - Histograms  
   - Correlation matrix

4. **Preprocessing**  
   - Missing value imputation (median)  
   - Feature scaling (standardization)

5. **Modeling**  
   - Pipelines integrating preprocessing and models  
   - 5-fold cross-validation  
   - Metric: RMSE

6. **Model Evaluation**  
   - Select best by CV RMSE  
   - Final evaluation on hold-out test set

## Project Structure

```

project/
├── data
├── data
      ├── processed
      └── raw
├── models
├── src
      ├── data
      ├── features
      ├── inference
      └── models
├── notebooks
├── tests
├── requirements.txt
└── README.md

```

## Configuration

Optional environment variables (with defaults):

- `RAW_DATA_PATH`: `data/raw`
- `PROCESSED_DATA_PATH`: `data/processed`
- `MODEL_DIR`: `models`
- `MODEL_FILENAME`: `best_pipeline.joblib`

## How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/Gefft3/regression-model-comparison.git
   cd regression-model-comparison
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download data:

   ```bash
   python -m src.data.download
   ```

4. Preprocess data:

   ```bash
   python -m src.data.preprocess
   ```

5. Train models:

   ```bash
   python -m src.models.train
   ```

6. Run inference:

   ```bash
   python -m src.inference.inference "<json_features>"
   ```

## Testing

Run unit tests:

```bash
pytest
```