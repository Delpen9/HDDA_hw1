import numpy as np
from scipy.interpolate import splrep, BSpline

def get_all_b_spline_coefficients(
    x : np.ndarray,
    y : np.ndarray,
    knot_count : int
) -> np.ndarray:
    '''
    b_spline():
        Perform a B-spline interpolation with knots as an input parameter and then output the coefficients of each BSpline
        x: array of x-coordinates of data points
        y: array of y-coordinates of data points
        knot_count: the number of knots expected for the spline object
    '''
    assert x.shape == y.shape
    assert x.shape[0] > knot_count + 6

    knot_distance = (np.amax(x) - np.amin(x)) / (knot_count - 1)
    knots = np.arange(np.amin(x), np.amax(x) + knot_distance, knot_distance)

    coefficients = np.zeros(shape = (knot_count + 6))
    for i in range(x.shape[0]):
        tck = splrep(x[i, :], y[i, :], t = knots[1: -1])
        spline = BSpline(*tck)
        if i == 0:
            coefficients = spline.c
        else:
            coefficients = np.vstack((
                coefficients,
                spline.c
            ))

    return coefficients
