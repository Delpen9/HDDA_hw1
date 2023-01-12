import numpy as np
from scipy.interpolate import UnivariateSpline

def compute_mse(x, y, spline):
    """
    Compute the mean squared error of a spline
    x: array of x-coordinates of data points
    y: array of y-coordinates of data points
    spline: spline object
    """
    y_pred = spline(x)
    mse = np.mean((y - y_pred)**2)
    return mse

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

# Create the spline
spline = UnivariateSpline(x, y, s=0.1)

mse = compute_mse(x, y, spline)
print("Mean Squared Error: ", mse)
