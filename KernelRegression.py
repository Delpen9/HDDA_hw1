import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression

def kernel_regression(x, y, kernel='rbf', optimal_lambda=None):
    """
    Perform kernel regression using a Gaussian kernel with an optimal lambda input parameter
    x: array of x-coordinates of data points
    y: array of y-coordinates of data points
    kernel: kernel to use in the regression (default: rbf)
    optimal_lambda: optimal lambda to use in the regression (default: None)
    """
    kr = KernelRidge(kernel=kernel)
    if optimal_lambda is None:
        param_grid = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]}
        grid = GridSearchCV(kr, param_grid, cv=5)
        grid.fit(x, y)
        kr = grid.best_estimator_
    else:
        kr.alpha = optimal_lambda
        kr.fit(x, y)
    return kr

# Generating a dataset for demonstration
x, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Using the default optimal lambda
kr = kernel_regression(x, y)
print("Optimal lambda (alpha) = ", kr.alpha)

# Using a specific lambda
optimal_lambda = 0.1
kr = kernel_regression(x, y, optimal_lambda=optimal_lambda)
print("Optimal lambda (alpha) = ", kr.alpha)
