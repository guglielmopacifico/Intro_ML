# ============================================================================

# INTRODUCTION TO MACHINE LEARNING
# TASK 1a MAIN CODE

# AUTHORS: G. PACIFICO, L. TORTORA, M. MELENNEC

# ============================================================================

# IMPORTS

import numpy as np

import pandas as pd
from pandas import DataFrame as df

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# ============================================================================

# VARIABLES

lbdas = [0.1, 1, 10, 100, 200]          # Regularisation parameters

n_folds = 10                            # Number of folds to make

rand = 42                               # Fix seed

train = 'task1a_do4bq81me/train.csv'    # Path to training data
errors = 'task1a_do4bq81me/errors.csv'  # Path to output document

# ============================================================================

# TOOLS

def get_data(path):
    """Extracts data from path and yields X, y data"""

    data = pd.read_csv(path)

    y = data.pop('y')
    y = y.to_numpy()
    X = data.to_numpy()

    return X, y

def rmse(y, y_pred):
    """Root Mean Squared Error of y_pred wrt y"""

    return mean_squared_error(y, y_pred)**0.5

def optimal(lbda, X, y):
    """Computes coefficients minimising the loss"""

    M = np.linalg.inv(X.T @ X + lbda)
    w = M @ (X.T @ y)
    return w

# ============================================================================

# WORK

X, y = get_data(train)          # Extract data from train.csv

# Generate K-fold model
kf = KFold(n_splits=n_folds, shuffle=True, random_state=rand)

err_lbda = []                   # Initialise mean errors array

for lbda in lbdas:              # Iterate over regularisation parameters

    # Define Ridge model for lbda
    model = Ridge(alpha=lbda, random_state=rand)

    err_k = []                  # Initialise error array for fixed lambda

    splits = kf.split(X)        # Get K-fold splitting indices generator

    # Iterate over folds
    for train_indices, test_indices in splits:

        # Split data into train and test folds
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        model.fit(X_train, y_train)             # Fit model to data
        yk_pred = model.predict(X_test)         # Get predicted y values
        err_k.append(rmse(y_test, yk_pred))     # Get RMSE

    err_lbda.append(np.mean(np.array(err_k)))

# ============================================================================

# POST-WORK

with open(errors, 'w') as file:
    for e in err_lbda: file.write(str(e)+'\n')