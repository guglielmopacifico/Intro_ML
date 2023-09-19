# ============================================================================

# INTRODUCTION TO MACHINE LEARNING
# TASK 1a MAIN CODE

# AUTHORS: G. PACIFICO, L. TORTORA, M. MELENNEC

# ============================================================================

# IMPORTS

from xml.sax import parseString
import pandas as pd
from pandas import DataFrame as df

import math

import numpy as np
from numpy.linalg import norm


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ============================================================================

# CONSTANTS


## REGRESSION PARAMETERS

# Regularisation parameters
lbdas = [0.1, 1, 10, 100, 200]
# Random state to shuffle data for the solver
rand = 10

## CROSS-VALIDATION CONSTANTS

# Number of folds
K = 10
# Test set proportion
test_prop = 0.1


## DATA-ACCESS

# Path to csv data files
train = 'task1a_do4bq81me/train.csv'
prediction = 'task1a_do4bq81me/prediction.csv'


# ============================================================================

# TOOLS


## CROSS-VALIDATION

def splitting(D, K, test_prop):
    """
    This function splits the dataset for cross-validation and testing
    Inputs:
        - D: Data set as numpy array
        - K: Number of folds
        - test_prop: Proportion of the data set to be used for testing
    Outputs:
        - D_prime: List of K subsets D'_i for cross-validation as numpy arrays
        - D_2prime: test subset as numpy array
    """

    N = len(D)
    n_test = int(test_prop*N)

    # Split D_prime in K subsets of almost equal length
    D_prime = D[:-n_test].copy()
    D_prime = np.array_split(D_prime, K)

    # Define D_2prime
    D_2prime = D[-n_test:].copy()

    return D_prime, D_2prime


## DATA ACCESS

# Define data extraction tool
def extraction(path):
    """
    This function extracts data from csv files
    Inputs:
        - path: string containing path to the data to be extracted
    Outputs:
        - X: training input values as numpy array of shape (len(data),13)
        - y: training output values as numpy array of length (len(data))
    """

    # Create data DataFrame
    data = pd.read_csv(path)

    # Split the data as input and output X, y (resp.)
    y = data.pop('y')
    y = y.to_numpy()
    X = data.to_numpy()

    return X, y

# Define data writing tool
def writing(path, errors):
    """
    This function writes the root mean squared errors obtained for 
    the different regularisation parameters in a csv file
    Inputs:
        - path: string containing the path to the file to write
        - errors: errors obtained for lbda1 to lbda5 (ordered) as a list 
            of floating point values
    Outputs:
        None
    """

    with open(path, 'w') as file:
        for e in errors: file.write(str(e)+'\n')


## REGRESSION

# Define regression function
def regression(model, X, y):
    """
    This function returns the regression coefficients for y = f(X) by
    minimizing the precised model
    Inputs:
        - model: model to be used for regression
        - X: array of input values
        - y: array of output values
    Outputs:
        - coeff: coefficients of regression y = f(X) for model
    """
    Xnan = np.isnan(np.min(X))
    ynan = np.isnan(np.min(y))

    model.fit(X, y)

    return model.coef_


## ERRORS

# Define root mean squared error
def rmse(y, y_pred):
    """Root mean squared error of y_pred wrt y"""
    return mean_squared_error(y, y_pred)**0.5
    

# ============================================================================

# WORK


## DATA EXTRACTION AND TREATMENT

# Extract data
X_train, y_train = extraction(train)

# Divide data as required
DX_prime, DX_2prime = splitting(X_train, K, test_prop)
Dy_prime, Dy_2prime = splitting(y_train, K, test_prop)


## GET RMSE FOR REGULARISATION PARAMETERS

# Initiate average errors:
avg_err = []

# Iterate over regularisation parameters
for lbda in lbdas:

    # Initiate ridge redression model with stochastic average gradient descent
    model = Ridge(alpha=lbda, solver='sag', random_state=rand)

    # Initiate validation errors:
    R = []

    # Iterate over folds
    for k in range(K):
        print(lbda, k)
        
        # Initiate D_k as dummy array with needed shape
        Dk_X = np.empty((1, X_train.shape[1]))
        Dk_y = np.empty(1)
        # Concatenate training subsets
        for i in range(K):
            if i != k:
                Dk_X = np.concatenate((Dk_X, DX_prime[i]))
                Dk_y = np.concatenate((Dk_y, Dy_prime[i]))
        
        # Remove dummy entries from D_k
        np.delete(Dk_X, 0, 0)
        np.delete(Dk_y, 0)

        # Train on D_k
        w_k = regression(model, Dk_X, Dk_y)

        # Compute validation error
        R_k = rmse(Dy_prime[k], np.dot(DX_prime[k], w_k))
        R.append(R_k)
    
    # Compute average error
    average = np.mean(R)
    avg_err.append(average)

# Save errors in csv file
writing(prediction, avg_err)
    


    
