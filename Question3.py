import os

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from scipy.interpolate import splrep, BSpline
from scipy.interpolate import UnivariateSpline
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

def k_fold_cross_validation_kernel_regression(
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
                    spline = KernelRidge(kernel = 'rbf',  alpha = lambda_value)
                    spline.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

                    y_pred = spline.predict(x_test.reshape(-1, 1))
                    error = mean_squared_error(y_test, y_pred.flatten())
                    errors.append(error)
                except:
                    errors.append(np.inf)
            lambda_errors.append(np.mean(errors))
        best_lambda_per_spline.append(lambdas[np.argmin(lambda_errors)])

    best_lambda = np.mean(best_lambda_per_spline)

    return best_lambda

def kernel_regression(
    x : np.ndarray,
    y : np.ndarray
) -> object:
    """
    kernel_regression():
        Perform kernel regression using a Gaussian kernel with an optimal lambda input parameter
        x: array of x-coordinates of data points
        y: array of y-coordinates of data points
    """
    assert x.shape == y.shape

    lambdas = np.arange(0.1, 5, (5 - 0.1) / 50)
    optimal_lambda = k_fold_cross_validation_kernel_regression(x, y, lambdas, 10)

    spline = KernelRidge(kernel = 'rbf',  alpha = optimal_lambda)
    spline.fit(x[0, :].reshape(-1, 1), y[0, :].reshape(-1, 1))

    return spline
def k_fold_cross_validation_smoothing_spline(
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
    optimal_lambda = k_fold_cross_validation_smoothing_spline(x, y, lambdas, 5)

    spline = UnivariateSpline(x[0, :], y[0, :], s = optimal_lambda)

    return spline

def cubic_spline(
    x : np.ndarray,
    y : np.ndarray,
    knot_count : int
) -> object:
    '''
    cubic_spline():
        Perform a cubic interpolation with knots as an input parameter
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
        tck = splrep(x[i, :], y[i, :], k = 3, t = knots[1: -1])
        spline = BSpline(*tck)
        coefficients = np.add(
            coefficients,
            spline.c[:knot_count + 6]
        )

    coefficients /= x.shape[0]

    control_points = tck[0]
    spline = BSpline(t = control_points, c = coefficients, k = 3)

    return spline

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

def compute_mse(
    x : np.ndarray,
    y : np.ndarray,
    spline : object
) -> float:
    """
    compute_mse():
        Compute the mean squared error of a spline
        x: array of x-coordinates of data points
        y: array of y-coordinates of data points
        spline: spline object
    """
    if isinstance(spline, KernelRidge):
        y_pred = spline.predict(x.reshape(-1, 1)).flatten()
    else:
        y_pred = spline(x)

    error = mean_squared_error(y, y_pred)

    return error

def plot_mean_function(
  x : np.ndarray,
  y : np.ndarray,
  spline : object,
  subfolder : str,
  filename : str
) -> None:
  '''
  plot_mean_function():
    
  '''
  # Plot the sample average signal
  plt.scatter(x, y, s = 1, label = 'Sample Average Signal')

  # Plot the estimated mean function
  if isinstance(spline, KernelRidge):
    plt.plot(x, spline.predict(x.reshape(-1, 1)), label = 'Estimated Mean Function', color = 'red')
  else:
    plt.plot(x, spline(x), label = 'Estimated Mean Function', color = 'red')
  plt.legend()
  
  plt.xlim(np.amin(x), np.amax(x))
  plt.ylim(np.amin(y), np.amax(y))

  if not os.path.exists(subfolder):
    os.makedirs(subfolder)

  plt.savefig(fr'{subfolder}/{filename}')
  plt.clf()

def cubic_spline_chart(
  data : pd.DataFrame
):
  '''
  cubic_spline_chart():
  '''
  spline_x = np.tile(
    np.arange(0, data.shape[1]),
    (data.shape[0], 1)
  )
  spline_y = data.to_numpy()

  cubic_spline_output = cubic_spline(spline_x, spline_y, 8)

  X = np.repeat(
    a = np.arange(0, data.shape[1]),
    repeats = data.shape[0]
  ).astype(int)

  y = data.to_numpy().T.flatten()

  plot_mean_function(X, y, cubic_spline_output, 'images', 'cubic_spline_mean_function.png')

  mse = compute_mse(X, y, cubic_spline_output)
  return mse

def b_spline_chart(
  data : pd.DataFrame
):
  '''
  b_spline_chart():
  '''
  spline_x = np.tile(
    np.arange(0, data.shape[1]),
    (data.shape[0], 1)
  )
  spline_y = data.to_numpy()

  b_spline_output = b_spline(spline_x, spline_y, 8)

  X = np.repeat(
    a = np.arange(0, data.shape[1]),
    repeats = data.shape[0]
  ).astype(int)

  y = data.to_numpy().T.flatten()

  plot_mean_function(X, y, b_spline_output, 'images', 'b_spline_mean_function.png')

  mse = compute_mse(X, y, b_spline_output)
  return mse

def smoothing_spline_chart(
  data : pd.DataFrame
):
  '''
  smoothing_spline_chart():
  '''
  spline_x = np.tile(
    np.arange(0, data.shape[1]),
    (data.shape[0], 1)
  )
  spline_y = data.to_numpy()

  smoothing_spline_output = smoothing_spline(spline_x, spline_y)

  X = np.repeat(
    a = np.arange(0, data.shape[1]),
    repeats = data.shape[0]
  ).astype(int)

  y = data.to_numpy().T.flatten()

  plot_mean_function(X, y, smoothing_spline_output, 'images', 'smoothing_spline_mean_function.png')

  mse = compute_mse(X, y, smoothing_spline_output)
  return mse

def kernel_regression_spline_chart(
  data : pd.DataFrame
):
  '''
  kernel_regression_spline_chart():
  '''
  spline_x = np.tile(
    np.arange(0, data.shape[1]),
    (data.shape[0], 1)
  )
  spline_y = data.to_numpy()

  kernel_regression_spline_output = kernel_regression(spline_x, spline_y)

  X = np.repeat(
    a = np.arange(0, data.shape[1]),
    repeats = data.shape[0]
  ).astype(int)

  y = data.to_numpy().T.flatten()

  plot_mean_function(X, y, kernel_regression_spline_output, 'images', 'kernel_regression_spline_mean_function.png')

  mse = compute_mse(X, y, kernel_regression_spline_output)
  return mse

if __name__ == '__main__':
  working_directory = os.getcwd()
  file_name = 'X1-1.csv'
  file_path = os.path.join(working_directory, file_name)
  
  data = pd.read_csv(file_path)

  cubic_spline_mse = cubic_spline_chart(data)
  b_spline_mse = b_spline_chart(data)
  smoothing_spline_mse = smoothing_spline_chart(data)
  kernel_regression_mse = kernel_regression_spline_chart(data)

  spline_types = np.array(['cubic_spline', 'b_spline', 'smoothing_spline', 'kernel_regression']).astype(str).reshape(-1, 1)
  spline_errors = np.array([
    cubic_spline_mse,
    b_spline_mse,
    smoothing_spline_mse,
    kernel_regression_mse
  ]).astype(str).reshape(-1, 1)

  spline_performance_comparison = pd.DataFrame(np.hstack((spline_types, spline_errors)), columns = ['Spline Type', 'Error'])
  performance_path = os.path.join(working_directory, 'spline_performance_comparison.csv')
  spline_performance_comparison.to_csv(performance_path, index = False)