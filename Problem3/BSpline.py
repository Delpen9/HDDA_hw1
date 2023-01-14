import numpy as np
from scipy.interpolate import splrep, BSpline

def b_spline(
    x : np.ndarray,
    y : np.ndarray,
    knot_count : int
) -> object:
    '''
    b_spline():
        Perform a B-spline interpolation with knots as an input parameter
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
        coefficients = np.add(
            coefficients,
            spline.c
        )

    coefficients /= x.shape[0]

    control_points = tck[0]
    degree = tck[2]
    spline = BSpline(t = control_points, c = coefficients, k = degree)

    return spline
