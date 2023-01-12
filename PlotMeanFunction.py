import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

# Generate some data
x = np.linspace(0, 10, num=100)
y = np.sin(x) + np.random.normal(0, 0.2, size=100)

# Create the spline
spline = UnivariateSpline(x, y, s=0.1)

# Plot the sample average signal
plt.plot(x, y, label='Sample Average Signal')

# Plot the estimated mean function
plt.plot(x, spline(x), label='Estimated Mean Function', color='red')
plt.legend()
plt.show()
