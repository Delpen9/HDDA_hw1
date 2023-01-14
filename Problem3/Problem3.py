from BSpline import b_spline
from CubicSpline import cubic_spline
from KernelRegression import kernel_regression
from SmoothingSpline import smoothing_spline

from ComputeMSE import compute_mse
from PlotMeanFunction import plot_mean_function

import os

import pandas as pd

if __name__ == '__main__':
  working_directory = os.getcwd()
  file_name = 'X1-1.csv'
  file_path = os.path.join(working_directory, file_name)
  
  data = pd.read_csv(file_path)
  X = data.iloc[:, :-1].to_numpy().astype(float)
  y = data.iloc[:, -1].to_numpy().astype(float)

  b_spline = b_spline(X, y, 8, 3)
  plot_mean_function(X, y, b_spline)