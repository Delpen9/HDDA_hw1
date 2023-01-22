# Standard Libraries
import numpy as np
import pandas as pd
import os

# Data Pre-processing
from scipy.interpolate import splrep, BSpline
from sklearn.decomposition import PCA

# Modeling
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier

# Performance evaluation
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_all_b_spline_coefficients(
    x : np.ndarray,
    y : np.ndarray,
    knot_count : int
) -> np.ndarray:
    '''
    b_spline():
        Perform a B-spline interpolation with knots as an input parameter and then output the coefficients of each BSpline
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
        if i == 0:
            coefficients = spline.c
        else:
            coefficients = np.vstack((
                coefficients,
                spline.c
            ))

    return coefficients

def transform_data(
  csv_file : str,
  knot_count : int
):
  # Load the data from CSV file
  df = pd.read_csv(csv_file)

  # Split the data into features and target
  X = df.iloc[:, 1:]
  y = df.iloc[:, 0].to_numpy().reshape(-1, 1)
  y = np.where(y == -1, 0, 1)

  spline_x = np.tile(
    np.arange(0, X.shape[1]),
    (X.shape[0], 1)
  )
  spline_y = X.to_numpy()

  # Perform B-spline transformation
  bspline_coefficients = get_all_b_spline_coefficients(spline_x, spline_y, knot_count)

  # Perform FPCA: Keep in mind that fPCA is simply PCA but on time-series data
  pca = PCA(n_components = 1)
  X_pca = pca.fit_transform(bspline_coefficients)
  return X_pca, y

def spline_fpca_training(
  csv_file : str,
  knot_count : int,
  k : int
) -> object:
  '''
  spline_fpca_prediction():
  '''
  X_pca, y = transform_data(csv_file, knot_count)

  # Initialize the XGBoost model
  xgb = XGBClassifier()

  # Initialize RandomizedSearchCV
  param_dist = {
      'n_estimators': [50, 100, 200],
      'max_depth': [2, 4, 6],
      'learning_rate': [0.1, 0.2, 0.3],
      'subsample': [0.5, 0.7, 1.0],
      'colsample_bytree': [0.5, 0.7, 1.0],
      'colsample_bylevel': [0.5, 0.7, 1.0],
      'min_child_weight': [1, 2, 4],
      'reg_alpha': [0, 0.1, 1],
      'reg_lambda': [0, 0.1, 1],
      'scale_pos_weight': [1, 2, 4]
  }
  random_search = RandomizedSearchCV(xgb, param_distributions = param_dist, n_iter = 10, cv = k)
  random_search.fit(X_pca, y)
  
  return random_search

if __name__ == '__main__':
  working_directory = os.getcwd()

  TRAINING_SET = 'ECG200TRAIN.csv'
  TRAINING_SET_PATH = os.path.join(working_directory, TRAINING_SET)

  KNOT_COUNT = 8
  k = 5

  best_model = spline_fpca_training(TRAINING_SET_PATH, KNOT_COUNT, k)

  TESTING_SET = 'ECG200TEST.csv'
  TESTING_SET_PATH = os.path.join(working_directory, TESTING_SET)

  X_test, y_test = transform_data(TESTING_SET_PATH, KNOT_COUNT)

  y_prob = best_model.predict_proba(X_test)[:, 1]

  # Calculate the ROC AUC
  roc_auc = roc_auc_score(y_test, y_prob)

  # Get the fpr, tpr, and thresholds for the ROC curve
  fpr, tpr, thresholds = roc_curve(y_test, y_prob)

  # Plot the ROC curve
  plt.plot(fpr, tpr, label = 'ROC AUC: {:.3f}'.format(roc_auc))
  plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Random')

  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  plt.legend()

  image_path = os.path.join(working_directory, 'roc_auc_curve.png')

  plt.savefig(image_path)
  plt.clf()

  # Calculate the accuracy at different threshold values
  accuracy = []
  for threshold in np.arange(0, 0.95, 0.01):
      y_pred = [1 if probability > threshold else 0 for probability in y_prob]
      accuracy_at_threshold = accuracy_score(y_test, y_pred)
      accuracy.append(accuracy_at_threshold)

  plt.plot(np.arange(0, 0.95, 0.01), accuracy)
  
  plt.title('Accuracy of Model at Multiple Thresholds')
  plt.xlabel('Threshold')
  plt.ylabel('Accuracy')

  image_path = os.path.join(working_directory, 'accuracy_at_multiple_thresholds.png')

  plt.savefig(image_path, dpi = 300)
  plt.clf()

  # Calculate the f1-score at different threshold values
  f1 = []
  for threshold in np.arange(0, 0.95, 0.01):
      y_pred = [1 if probability > threshold else 0 for probability in y_prob]
      f1_at_threshold = f1_score(y_test, y_pred)
      f1.append(f1_at_threshold)

  plt.plot(np.arange(0, 0.95, 0.01), f1)
  
  plt.title('F1-Score of Model at Multiple Thresholds')
  plt.xlabel('Threshold')
  plt.ylabel('F1-Score')

  image_path = os.path.join(working_directory, 'f1_at_multiple_thresholds.png')

  plt.savefig(image_path, dpi = 300)
  plt.clf()

  selected_threshold = 0.9

  y_pred = [1 if probability > selected_threshold else 0 for probability in y_prob]

  confusion_matrix_values = confusion_matrix(y_test, y_pred)

  # Plot the confusion matrix using matplotlib
  sns.heatmap(confusion_matrix_values, annot = True, fmt = 'd', cmap = 'Blues')

  # Add labels to the plot
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')

  image_path = os.path.join(working_directory, 'confusion_matrix.png')

  plt.savefig(image_path, dpi = 300)
  plt.clf()
