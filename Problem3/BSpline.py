import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

def b_spline(
    x : np.ndarray,
    y : np.ndarray,
    knot_count : int,
    degree : int
) -> list[object]:
    '''
    b_spline():
        Perform a B-spline interpolation with knots as an input parameter
        x: array of x-coordinates of data points
        y: array of y-coordinates of data points
        knot_count: the number of knots expected for the spline object
        degree: degree of the B-spline
    '''
    splines = []
    for i in range(X.shape[1]):
        spl = make_interp_spline(np.arange(len(X)), X[:, i], k = degree, t = knot_count)
        spline = BSpline(*spl.get_knots(), spl.get_coeffs(), k = degree, extrapolate = False)
        splines.append(spline)

    return splines
