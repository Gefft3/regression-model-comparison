from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def get_models():
    """
    Get a dictionary of regression models for training.
    Returns:
        dict: A dictionary where keys are model names and values are model instances.
    """

    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "CatBoost": CatBoostRegressor(iterations=100, silent=True, random_seed=42),
    }
