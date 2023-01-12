import numpy as np
from scipy.interpolate import CubicSpline

def cubic_spline(x, y, knots):
    """
    Perform a cubic spline interpolation with knots as an input parameter
    x: array of x-coordinates of data points
    y: array of y-coordinates of data points
    knots: array of knots (breakpoints) to be used in the spline
    """
    spline = CubicSpline(x, y, bc_type='not-a-knot', knots=knots)
    return spline

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])
knots = np.array([2, 3, 4])

spline = cubic_spline(x, y, knots)

# Evaluate the spline at a specific point
x_eval = 3.5
y_eval = spline(x_eval)
print("y = ", y_eval, "at x = ", x_eval)
