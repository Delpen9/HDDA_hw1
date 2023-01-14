import numpy as np
from scipy.interpolate import CubicSpline

def cubic_spline(
    x : np.ndarray,
    y : np.ndarray,
    knots : np.ndarray
) -> object:
    '''
    cubic_spline():
        Perform a cubic spline interpolation with knots as an input parameter
        x: array of x-coordinates of data points
        y: array of y-coordinates of data points
        knots: array of knots (breakpoints) to be used in the spline
    '''
    spline = CubicSpline(x, y, bc_type = 'not-a-knot', knots = knots)
    return spline
