import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

def plot_mean_function(
  x : np.ndarray,
  y : np.ndarray,
  spline : object
) -> None:
  '''
  plot_mean_function():
    
  '''
  # Plot the sample average signal
  plt.plot(x, y, label = 'Sample Average Signal')

  # Plot the estimated mean function
  plt.plot(x, spline(x), label = 'Estimated Mean Function', color = 'red')
  plt.legend()
  plt.show()
