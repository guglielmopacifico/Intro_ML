# ============================================================================

# INTRODUCTION TO MACHINE LEARNING

# TASK 2 - MEDICAL EVENTS PREDICTION

# AUTHORS: M. Melennec, G. Pacifico, L. Tortora

# ============================================================================

# IMPORTS

import numpy as np

import pandas as pd
from pandas import DataFrame as df

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

from sklearn.svm import SVC


# ============================================================================

# VARIABLES

## Paths to data files

train = 'task2_k49am2lqi/imp_train_features.csv'        # Training features
train_lbl = 'task2_k49am2lqi/train_labels.csv'        # Training labels
test = 'test_features.csv'                              # Testing features

## Model variables

eps = 10**-7            # Tolerance for stopping criteria
k_max = -1              # Max number of iterations
cache = 200             # Kernel cache size (in MB)
seed = 42               # Random seed for regression

# ============================================================================

# TOOLS

def extract_feat_imp(path):
    """Extracts data from feature file in path."""

    data = pd.read_csv(path)

    x_train = data.to_numpy()

    return x_train


def extract_labels(path):
    """Extracts data from label file in path."""

    data = pd.read_csv(path)

    y = data.to_numpy()[:,1:11]     # Remove pid column and tasks 2,3 labels

    return y


def feat_transform(data):
    """Transforms the features to a list, for all measurment types
    of the mean, median, variance, min and max."""

    num_patient = int(data.shape[0]/12)     # Number of patients in sample
    num_mmt = int(data.shape[1])            # Number of measurments

    new_data = np.empty((num_patient, 5*num_mmt))

    for i in range(num_patient):                # Iterate over patients
        patient_data = data[i*12:(i+1)*12]      # Isolate patient i's data
        
        # # Initialise new data array for patient i
        # new_patient_data = np.array([])

        # Deal with the age
        age = patient_data[0,0]
        age_new_feat = [age, age, 0, age, age]
        new_data[i, :5] = np.array(age_new_feat)
        
        for j in range(1, num_mmt):             # Iterate over labels
            values = patient_data[:,j]          # Extract measurment j values

            # Compute new features
            mean = np.mean(values)
            median = np.median(values)
            var = np.var(values)
            min = np.min(values)
            max = np.max(values)

            # Save new features
            new_data[i, 5*j:5*(j+1)] = np.array([mean, median, var, min, max])

    return new_data

# ============================================================================

# WORK

## Extract data

train_feat = extract_feat_imp(train)
labels = extract_labels(train_lbl)[:700]

## Pre-processing of the data
trans_feat = feat_transform(train_feat)
# Scale data
scaler = StandardScaler().fit(trans_feat)
scaled_trans_feat = scaler.transform(trans_feat)

# ## Set up he SVM
# svm = SVC(C=1.0, kernel='linear', tol=eps, cache_size=cache,
#             random_state=seed)
# # CHOOSE C THROUGH KFOLD VALIDAION?

## Fit data
fitted = dict()                 # Dict to save fitted models

# Iterate over needed labels
for l in range(labels.shape[1]):
    # Initiate a new SVM
    svm = SVC(C=1.0, kernel='linear', tol=eps, cache_size=cache,
                random_state=seed)
    svm.fit(scaled_trans_feat, labels[:,l])     # Fit SVM

    fitted[l] = svm                             # Save fiitted SVM















## Pre-processing of the data

# scaler = StandardScaler().fit(train_feat)
# scaled_feat = scaler.transform(train_feat)

# scaled_new_train = feat_transform(scaled_feat)

# new_train = feat_transform(train_feat)      # Feature transformation

# # Scale data for convergence of stochastic avg gradient descent
# scaler = StandardScaler().fit(new_train)
# scaled_new_train = scaler.transform(new_train)


# ## Initialise and fit the model
# model_lr = LogisticRegression(penalty='l2', tol=eps, fit_intercept=True, 
#             random_state=seed, solver='sag', max_iter=k_max,
#             multi_class='ovr')

# # model_rc = RidgeClassifier()

# fitted = dict()         # Initialise dict of fitted models for each label

# for l in range(labels.shape[1]):        # Iterate over labels
#     fitted[l] = model_lr.fit(scaled_new_train, labels[:,l])


## Test fitted model

