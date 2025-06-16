# Regression Model Comparison Pipeline

This repository presents a complete and well-documented machine learning pipeline for comparing regression models using the California Housing dataset.

## Objective

Predict the median house value based on socioeconomic and geographic features using:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* CatBoost Regressor

## Motivation

Choosing the right regression model is critical for achieving accurate predictions. Each model has unique strengths and weaknesses depending on the dataset's characteristics, such as feature interactions, non-linear relationships, and noise. By testing multiple models, we can identify the one that best balances bias and variance for this specific problem, ensuring robust and reliable predictions.

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
regression-model-comparison-pipeline/
├── data
│   ├── processed
│   └── raw
├── models
├── notebooks
├── plots
├── src
│   ├── data
│   ├── features
│   ├── inference
│   └── models
├── tests
├── api
├── Dockerfile
├── requirements.txt
└── README.md
```

## Configuration

Optional environment variables (with defaults):

- `BASE_DIR`: `root`
- `RAW_DATA_PATH`: `data/raw`
- `PROCESSED_DATA_PATH`: `data/processed`
- `MODEL_DIR`: `models`
- `MODEL_FILENAME`: `best_pipeline.joblib`
- `PLOTS_DIR`: `plots`

## How to Run

### Using Docker Compose

1. Build the Docker image (first time):

   ```bash
   docker-compose up --build
   ```

2. Run the container (not first time):

   ```bash
   docker-compose up
   ```

3. Enter the container to run the commands:
   ```bash
   docker exec -it regression-model-comparison-app-1 sh
   ```

4. Download and preprocess data (inside the container):

   ```bash
   python3 -m src.data.download
   python3 -m src.data.preprocess
   ```

5. Train models (inside the container):

   ```bash
   python3 -m src.models.train
   ```

6. Run inference (inside the container):

   ```bash
   python3 -m src.inference.inference "<json_features>"
   ```

7. Access the API for inference:

   Send a POST request to `http://{API_HOST}:{API_PORT}/predict` with the input features in JSON format.

## Testing

Run unit tests (inside the container):

```bash
pytest
```

## Next Steps
The next steps could include:
- Hyperparameter tuning for the best model
- Monitoring model performance over time
- Exploring more complex models or ensembles