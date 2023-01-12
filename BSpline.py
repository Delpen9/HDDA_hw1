import numpy as np
from scipy.interpolate import BSpline

def b_spline(x, y, knots, degree):
    """
    Perform a B-spline interpolation with knots as an input parameter
    x: array of x-coordinates of data points
    y: array of y-coordinates of data points
    knots: array of knots (breakpoints) to be used in the spline
    degree: degree of the B-spline
    """
    spline = BSpline(knots, y, degree)
    return spline

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])
knots = np.array([1, 2, 3, 4, 5])
degree = 3

spline = b_spline(x, y, knots, degree)

# Evaluate the spline at a specific point
x_eval = 3.5
y_eval = spline(x_eval)
print("y = ", y_eval, "at x = ", x_eval)
