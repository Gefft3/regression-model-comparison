from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def numeric_pipeline():
    """
    Create a preprocessing pipeline for numeric features.
    Returns:
        Pipeline: A scikit-learn pipeline that imputes missing values with the median
                  and scales features using StandardScaler.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
