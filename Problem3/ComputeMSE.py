import numpy as np

def compute_mse(
    x : np.ndarray,
    y : np.ndarray,
    spline : callable
) -> float:
    """
    compute_mse():
        Compute the mean squared error of a spline
        x: array of x-coordinates of data points
        y: array of y-coordinates of data points
        spline: spline object
    """
    y_pred = spline(x)
    mse = np.mean((y - y_pred)**2)
    return mse
