import numpy as np
from scipy.interpolate import UnivariateSpline

def smoothing_spline(
    x : np.ndarray,
    y : np.ndarray,
    smoothing_param : float
):
    '''
    smoothing_spline():
        Perform a smoothing spline interpolation with lambda as an input parameter
        x: array of x-coordinates of data points
        y: array of y-coordinates of data points
        smoothing_param: smoothing parameter (lambda)
    '''
    spline = UnivariateSpline(x, y, s=smoothing_param)
    return spline
