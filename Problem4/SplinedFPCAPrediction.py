import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.interpolate import BSpline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def spline_fpca_prediction(
  csv_file : str,
  degree : int,
  test_size : float,
  classifier : callable
) -> tuple[np.ndarray, np.ndarray]:
  '''
  spline_fpca_prediction():
  '''
  # Load the data from CSV file
  df = pd.read_csv(csv_file)

  # Split the data into features and target
  X = df.iloc[:, :-1]
  y = df.iloc[:, -1]

  # Define the knots and degree for the B-spline
  knots = np.linspace(X.min(), X.max(), num = 10)

  # Perform B-spline transformation
  bspline = BSpline(knots, np.array([0] * X.shape[1]), degree)
  X_bspline = bspline(X)

  # Perform FPCA
  pca = PCA()
  X_pca = pca.fit_transform(X_bspline)

  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = test_size)

  # Train the XGBoost model
  classifier_model = classifier()
  classifier_model.fit(X_train, y_train)

  # Evaluate the model on the test set
  prediction = classifier_model.predict(X_test)
  
  return prediction, y_test
