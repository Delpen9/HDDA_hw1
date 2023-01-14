import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge

import os

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

