import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import mean_squared_error

def k_fold_cross_validation(
    x : np.ndarray,
    y : np.ndarray,
    lambdas : np.ndarray,
    k : int
) -> dict:
    permutation = np.random.permutation(x.shape[1])

    x_copy = x[:, permutation].copy()
    y_copy = y[:, permutation].copy()

    best_lambda_per_spline = []
    for i in range(x_copy.shape[0]):

        lambda_errors = []
        for lambda_value in lambdas:
            x_folds = np.array([val[:int(np.floor(x.shape[1] / k))] for val in np.array_split(x_copy[i, :], k)])
            y_folds = np.array([val[:int(np.floor(x.shape[1] / k))] for val in np.array_split(y_copy[i, :], k)])

            errors = []
            for fold in range(k):
                positions = np.delete(np.arange(0, k), fold).astype(int)

                x_train = x_folds[positions].flatten()
                y_train = y_folds[positions].flatten()

                indices = np.argsort(x_train)
                x_train = x_train[indices]
                y_train = y_train[indices]

                x_test = x_folds[fold]
                y_test = y_folds[fold]
                
                try:
                    spline = UnivariateSpline(x_train, y_train, s = lambda_value)

                    y_pred = spline(x_test)
                    error = mean_squared_error(y_test, y_pred)
                    errors.append(error)
                except:
                    errors.append(np.inf)
            lambda_errors.append(np.mean(errors))
        best_lambda_per_spline.append(lambdas[np.argmin(lambda_errors)])

    best_lambda = np.mean(best_lambda_per_spline)

    return best_lambda

def smoothing_spline(
    x : np.ndarray,
    y : np.ndarray
) -> object:
    '''
    smoothing_spline():
        Perform a smoothing spline interpolation with lambda as an input parameter
        x: array of x-coordinates of data points
        y: array of y-coordinates of data points
    '''
    assert x.shape == y.shape

    lambdas = np.arange(0.1, 5, (5 - 0.1) / 50)
    optimal_lambda = k_fold_cross_validation(x, y, lambdas, 5)

    spline = UnivariateSpline(x[0, :], y[0, :], s = optimal_lambda)

    return spline
