from BSpline import b_spline
from CubicSpline import cubic_spline
from KernelRegression import kernel_regression
from SmoothingSpline import smoothing_spline

from ComputeMSE import compute_mse
from PlotMeanFunction import plot_mean_function

import os

if __name__ == '__main__':
  working_directory = os.getcwd()
  file_name = 'X1-1.csv'
  file_path = os.path.join(working_directory, file_name)
  
  data = df.read_csv(file_path)
  X = data.loc[:, :-1]
  y = data.loc[:, -1]
  
  
  
  

