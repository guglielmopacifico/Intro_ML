# ============================================================================

# INTRODUCTION TO MACHINE LEARNING

# TASK 2 - MEDICAL EVENTS PREDICTION
# MAIN FILE

# AUTHORS: M. Melennec, G. Pacifico, L. Tortora

# ============================================================================

# IMPORTS

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import imputation
import measurments
import sepsis
import predict

# ============================================================================

# VARIABLES

## Paths do data

train_feat = 'task2_k49am2lqi/train_features.csv'
train_labels = 'task2_k49am2lqi/train_labels.csv'
prediction = 'task2_k49am2lqi/predictions.csv'

## Imputation variables

k = 30          # Value for k nearest neighbours imputation

## Sub-task 1 variables

eps = 10**-7            # Tolerance for stopping criteria
k_max = -1              # Max number of iterations
cache = 200             # Kernel cache size (in MB)
seed = 42               # Random seed for regression

# ============================================================================

# TOOLS

def extract_labels(path):
    """Extracts data from label file in path."""

    data = pd.read_csv(path)

    y = data.to_numpy()[:,1:11]     # Remove pid column and tasks 2,3 labels

    return y

def feat_transform(data):
    """Transforms the features to a list, for all measurment types
    of the mean, median, variance, min and max 
    and scale the transformed features."""

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
    
    # Scale data
    scaler = StandardScaler().fit(new_data)
    scaled_data = scaler.transform(new_data)

    return scaled_data

# ============================================================================

# DATA PRE-PROCESSING

train_feat_imp = imputation.main()     # Impute data
train_feat_trans = feat_transform(train_feat_imp)
# test_feat = train_feat_trans[:,2:].copy()
train_feat_imp = train_feat_trans[:,2:].copy()   # Remove pid and time stamps

train_lbl = extract_labels(train_labels)

svm_mmt = measurments.main()    # Import subtask 1 fitted multilabel svm
svm_sepsis = sepsis.main()      # Import subtask 2 fitted svm

predict.main()

# test_feat = measurments.feat_transform(test_feat)

# test_pred = svm_mmt.predict(test_feat)
# test_lbl = pd.read_csv(train_labels).to_numpy()[:, 1:11]

# diff = np.equal(test_lbl,test_pred)

# diff_df = pd.DataFrame(diff)

# diff_df.to_csv('task2_k49am2lqi/test.csv')