# ============================================================================

# INTRODUCTION TO MACHINE LEARNING
# TASK 1b MAIN CODE

# AUTHORS: G. PACIFICO, L. TORTORA, M. MELENNEC

# ============================================================================

# IMPORTS

import numpy as np

import pandas as pd
from pandas import DataFrame as df

from sklearn.linear_model import LinearRegression as lr 
from sklearn.metrics import mean_squared_error

# ============================================================================

# DATA ACCESS

train = './train.csv'    # Path to training data
coeffs = './coeffs.csv'  # Path to output file

# ============================================================================

# TOOLS

def get_data(path):
    """Extracts data from path and yields X, y data"""

    data = pd.read_csv(path)
    data = data.drop('Id', axis = 1)

    y = data.pop('y')
    y = y.to_numpy()
    X = data.to_numpy()

    return X, y

def rmse(y, y_pred):
    """Root Mean Squared Error of y_pred wrt y"""

    return mean_squared_error(y, y_pred)**0.5

def features(X):
    """Builds featured array"""

    phi = np.ones((X.shape[0], 21))       # Initiate empty featured array

    phi[:, :5] = X[:]                     # phi1 to phi5
    phi[:, 5:10] = np.square(X[:])        # phi6 to phi10
    phi[:, 10:15] = np.exp(X[:])          # phi11 to phi15
    phi[:, 15:20] = np.cos(X[:])          # phi16 to phi20

    return phi

def optimal(X, y):
    """Computes coefficients minimising the loss"""

    M = np.linalg.inv(X.T @ X)
    w = M @ (X.T @ y)
    return w

# ============================================================================

# WORK

X, y = get_data(train)          # Extract data from train.csv

phi = features(X)

model = lr(fit_intercept=False, normalize=False)    # Define regression model
model.fit(phi, y)                                   # Fit featured data

w = model.coef_
w_star = optimal(phi, y)

print(rmse(w_star, w))

# ============================================================================

# POST-WORK

with open(coeffs, 'w') as file:
    for e in w: file.write(str(e)+'\n')