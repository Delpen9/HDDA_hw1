import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.interpolate import BSpline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load the data from CSV file
df = pd.read_csv("data.csv")

# Split the data into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Define the knots and degree for the B-spline
knots = np.linspace(X.min(), X.max(), num=10)
degree = 3

# Perform B-spline transformation
bspline = BSpline(knots, np.array([0]*X.shape[1]), degree)
X_bspline = bspline(X)

# Perform FPCA
pca = PCA()
X_pca = pca.fit_transform(X_bspline)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)

# Train the XGBoost model
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = xgb.score(X_test, y_test)
print("Accuracy: ", accuracy)
