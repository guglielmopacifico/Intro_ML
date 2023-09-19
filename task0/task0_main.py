# ============================================================================

# INTRODUCTION TO MACHINE LEARNING
# TASK 0 MAIN CODE

# AUTHORS: L. TORTORA & M. MELENNEC
# DATA CODE: sl19d1

# ============================================================================

# IMPORTS

import pandas as pd
from pandas import DataFrame as df

import math

import numpy as np
from numpy.linalg import norm

from sklearn.metrics import mean_squared_error


# ============================================================================

# CONSTANTS

# Path to csv data files
train = 'task0_sl19d1/train.csv'
test = 'task0_sl19d1/test.csv'
prediction = 'task0_sl19d1/prediction.csv'

# Output data column labels
labels = ['y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']

#Learning rate
eta = 10**-10
# Stopping criterion
eps = 10**-5
# Initial value
w0 = 0.9 * np.ones(10)


# ============================================================================

# TOOLS AND FUNCTIONS

# Define evaluation metric
RMSE = lambda y,y_pred: mean_squared_error(y, y_pred)**0.5


# Define gradient
def grad_sqrd(w, X, y):
    
    XT = np.transpose(X)
    grad = 2 * (np.dot(XT @ X, w) - np.dot(XT, y)) / y.shape[0]

    return grad


# Define gradient descent method
def grad_descent(gradient, initial, X, y, rate, threshold):

    w_now = initial + 2*threshold
    w_next = initial
    while norm(w_next - w_now) > threshold:
        w_now = w_next
        w_next = w_now - rate * gradient(w_now, X, y)
    
    return w_next


# Write csv with chosen data
def write_to_csv(path, labels, dataframe):

    dataframe.columns = labels
    dataframe.to_csv(path, index_label='Id')


# ============================================================================

# WORK

# Extract train data
train_df = pd.read_csv(train)

# Training
X_train = train_df[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']]
X_train = X_train.to_numpy()

y_train = train_df['y'].to_numpy()

# Check that eta allows convergence
print('Contraction factor rho =', np.linalg.norm(np.identity(10) - \
      eta*np.matmul(np.transpose(X_train),X_train), 2))

w_final = grad_descent(grad_sqrd, w0, X_train, y_train, eta, eps)


# Extract test data
test_df = pd.read_csv(test)

X_test = test_df[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']]
X = X_test.to_numpy()

y_test = np.dot(X, w_final)

# Print error
y_theo = np.mean(X_test, axis=1)
print('root mean squared error:', RMSE(y_theo, y_test))


# Write output CSV
X_test_df = df(X_test)
X_test_df.insert(loc=0, column='', value=y_test)
write_to_csv(prediction, labels, X_test_df)