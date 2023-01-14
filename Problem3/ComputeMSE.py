import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge

def compute_mse(
    x : np.ndarray,
    y : np.ndarray,
    spline : object
) -> float:
    """
    compute_mse():
        Compute the mean squared error of a spline
        x: array of x-coordinates of data points
        y: array of y-coordinates of data points
        spline: spline object
    """
    if isinstance(spline, KernelRidge):
        y_pred = spline.predict(x.reshape(-1, 1)).flatten()
    else:
        y_pred = spline(x)

    error = mean_squared_error(y, y_pred)

    return error