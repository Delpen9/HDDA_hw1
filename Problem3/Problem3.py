from BSpline import b_spline
from CubicSpline import cubic_spline
from KernelRegression import kernel_regression
from SmoothingSpline import smoothing_spline

from ComputeMSE import compute_mse
from PlotMeanFunction import plot_mean_function

import os

import pandas as pd
import numpy as np

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

  return None

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

  return None

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

  return None

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

  return None

if __name__ == '__main__':
  working_directory = os.getcwd()
  file_name = 'X1-1.csv'
  file_path = os.path.join(working_directory, file_name)
  
  data = pd.read_csv(file_path)

  cubic_spline_mse = cubic_spline_chart(data)
  b_spline_mse = b_spline_chart(data)
  smoothing_spline_mse = smoothing_spline_chart(data)
  kernel_regression_mse = kernel_regression_spline_chart(data)
