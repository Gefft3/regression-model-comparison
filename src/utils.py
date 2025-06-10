import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred, squared=False):
    """
    Calculate the Root Mean Squared Error (RMSE).

    Parameters:
    - y_true: array-like of shape (n_samples,) - Ground truth (correct) target values.
    - y_pred: array-like of shape (n_samples,) - Estimated target values.
    - squared: bool, default=False - If True, return Mean Squared Error (MSE) instead of RMSE.

    Returns:
    - float: RMSE or MSE value.
    """
    mse = mean_squared_error(y_true, y_pred)
    return mse if squared else np.sqrt(mse)
