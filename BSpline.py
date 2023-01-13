import numpy as np
from scipy.interpolate import BSpline

def b_spline(
    x : np.ndarray,
    y : np.ndarray,
    knots : np.ndarray,
    degree : int
) -> Object:
    '''
    b_spline():
        Perform a B-spline interpolation with knots as an input parameter
        x: array of x-coordinates of data points
        y: array of y-coordinates of data points
        knots: array of knots (breakpoints) to be used in the spline
        degree: degree of the B-spline
    '''
    spline = BSpline(knots, y, degree)
    return spline
    
