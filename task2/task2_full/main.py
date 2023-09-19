# ============================================================================

# INTRODUCTION TO MACHINE LEARNING

# TASK 2 - MEDICAL EVENTS PREDICTION
# MAIN FILE

# AUTHORS: M. Melennec, G. Pacifico, L. Tortora

# ============================================================================

# IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeCV

# ============================================================================

# VARIABLES

## Paths do data

dir = 'task2/' 

train_feat_path = dir + 'train_features.csv'
train_output_path = dir + 'train_labels.csv'
test_feat_path = dir + 'test_features.csv'
prediction_path = dir + 'predictions.csv'
trans_train_feat_path = dir + 'train_features_transformed.csv'
trans_test_feat_path = dir + 'test_features_transformed.csv'
test_pids_path = dir + 'tets_pids.csv'

## Feature selection variables

num_sel = 30    # Number of features to select for each training

## Training variables

# Sub-tasks 1&2
k_max = 50000       # Max number of iterations
seed = 42           # Random seed for regression

# Ridge regression regularisation parameters to choose upon (Sub-task 3)
lambdas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]

## Data pre-processing constants

transform_feats = True             # Whether or not to transform features

# Whether or not to save the input-output correlations
save_correlations_fig = False
correlations_path = dir + 'correlations.pdf'    # If yes, where?

# ============================================================================

# INTERNAL VARIABLES

## Labeling vectors

test_order = ['LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST',
                'LABEL_Alkalinephos','LABEL_Bilirubin_total',
                'LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2',
                'LABEL_Bilirubin_direct','LABEL_EtCO2']
sepsis = ['LABEL_Sepsis']
regression = ['LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']

# ============================================================================

# TOOLS

def extract_data(path, pids=False):
    """Extracts data, from path, of n many patients to be imputed."""

    data = pd.read_csv(path)

    data = data.to_numpy()  # Take the first "patients" patients

    return data

def extract_labels(path):
    """Extracts data from label file in path."""

    data = pd.read_csv(path)
    
    y = data.to_numpy()[:,1:]           # Remove pid column
    
    return y

def get_features(data, save_path=None):
    """Transforms the features to a list, for all measurment types
    of the mean, median, min, max and number of measurements of that type;
    does imputation via the global mean of the corresponding feature."""
    
    num_patient = int(data.shape[0]/12)     # Number of patients in sample
    num_mmt = int(data.shape[1])            # Number of measurements

    new_data = np.empty((num_patient, 6*num_mmt))

    for i in range(num_patient):                # Iterate over patients
        patient_data = data[i*12:(i+1)*12]      # Isolate patient i's data

        # Deal with the age
        age = patient_data[0,0]
        age_new_feat = [age, age, 0, age, age, 12]
        new_data[i, :6] = np.array(age_new_feat)
        
        for j in range(1, num_mmt):         # Iterate over labels
            values = patient_data[:,j]      # Extract values of j-th patient

            # Compute new features
            mean = np.nanmean(values)
            median = np.nanmedian(values)
            var = np.nanvar(values)
            min = np.nanmin(values)
            max = np.nanmax(values)
            nans = int(12 - np.sum(np.isnan(values)))

            # Save new features
            new_data[i, 6*j:6*(j+1)] =\
                np.array([mean, median, var, min, max, nans])
    
    # Means of each measurement used for imputation
    means_all = np.nanmean(new_data, axis=0)
    
    for i in range(num_patient):
        new_data[i][np.isnan(new_data[i])] = means_all[np.isnan(new_data[i])]

    new_data_df = pd.DataFrame(new_data)

    # Option to save data to be re-used on later run
    if save_path is not None:
        new_data_df.to_csv(save_path)
    
    return new_data_df

def select_features(X, Y, labeling, num_sel=num_sel, columns=False,
                    save_fig=False, fig_path=None):
    """Select num_sel features which are most correlated to
    the labeling output. Possibly save the correlations."""

    combined = pd.concat([X, Y],axis=1)
    corr = combined.corr()              # Correlation matrix

    # Only keep correlation of features to labels
    corr = corr.iloc[:X_train.shape[1],X_train.shape[1]:]

    if save_fig:
        # Output correlations
        fig = plt.figure(figsize=(20, 2))
        sns.heatmap(corr.T, square=True)
        plt.savefig(fig_path)

    if columns == False:
        best = corr[labeling].nlargest(n=num_sel)
    else: best = corr[labeling].nlargest(n=num_sel, columns='LABEL_Sepsis')

    best_indices = best.index

    return best_indices

# ============================================================================

# PRE-WORK

## Data extraction

# Get transformed features, either computing or using file
if transform_feats:
    # Training data
    train_feat = extract_data(train_feat_path)
    np.delete(train_feat, (0,1), axis=1)
    X_train =\
        get_features(train_feat, save_path=trans_train_feat_path)
    
    # Testing data
    test_feat = extract_data(test_feat_path)
    test_pids = pd.DataFrame(test_feat[::12,0].copy(), columns=['pid'])
    test_pids.to_csv(test_pids_path, columns=['pid'])             # Save pids
    np.delete(test_feat, (0,1), axis=1)
    X_test = get_features(test_feat, save_path=trans_test_feat_path)

else:
    X_train = pd.read_csv(trans_train_feat_path, index_col=0) # Training data
    X_test = pd.read_csv(trans_test_feat_path, index_col=0)   # Testing data
    test_pids = pd.read_csv(test_pids_path, index_col=0)

Y_train = extract_labels(train_output_path)     # Get training labels

print('Data extracted!')                        # Checkpoint

# ============================================================================

# WORK

## Sub-task 1

# Select training images for sub-task 1
Y = pd.DataFrame(Y_train[:, :10], columns=test_order)

st1_pred = pd.DataFrame(columns=test_order)     # Predictions DataFrame

for labeling in test_order:     # Iterate over test order types
    # Select features
    best_indices = select_features(X_train, Y, labeling)
    
    # Initialise SVM
    svm = SVC(random_state=seed, max_iter=k_max, class_weight='balanced', 
                probability=True)
    pipe = make_pipeline(StandardScaler(), svm)     # Learning pipeline

    pipe.fit(X_train[best_indices],Y[labeling])     # Fit SVM

    # Get real-velued prediction function
    f_pred = pipe.decision_function(X_test[best_indices])
    pred = 1/(1 + np.exp(-f_pred))

    # Save prediction function
    st1_pred[labeling] = pred               # Save prediction function

print('Sub-task 1 done!')          # Checkpoint


## Sub-task 2

# Select training images for sub-task 1
Y = pd.DataFrame(Y_train[:, 10], columns=sepsis)

st2_pred = pd.DataFrame(columns=sepsis)         # Predictions DataFrame

# Select features
best_indices = select_features(X_train, Y, sepsis, columns=True)

# Initialise SVM
svm = SVC(random_state=seed, max_iter=k_max, class_weight='balanced',
            probability=True)
pipe = make_pipeline(StandardScaler(), svm)     # Learning pipeline

pipe.fit(X_train[best_indices],Y[sepsis])       # Fit SVM

# Get real-velued prediction function
f_pred = pipe.decision_function(X_test[best_indices])
pred = 1/(1 + np.exp(-f_pred))

st2_pred['LABEL_Sepsis'] = pred                 # Save prediction function

print('Sub-task 2 done!')      # Chackpoint


## Sub-task 3

# Select training images for sub-task 1
Y = pd.DataFrame(Y_train[:, 11:], columns=regression)

st3_pred = pd.DataFrame(columns=regression)         # Predictions DataFrame

for labeling in regression:
    # Select features
    best_indices = select_features(X_train, Y, labeling)
    
    # Initialise SVM
    model = RidgeCV(alphas=lambdas)
    pipe = make_pipeline(StandardScaler(), model)       # Learning pipeline

    pipe.fit(X_train[best_indices],Y[labeling])         # Fit SVM

    # Get real-velued prediction function
    f_pred = pipe.predict(X_test[best_indices])

    st3_pred[labeling] = pred               # Save prediction function

print('Sub-task 3 done!')                   # Checkpoint


## Output predictions

# Concatenate all predictions into one
prediction = pd.concat([test_pids, st1_pred, st2_pred, st3_pred], axis=1)
prediction = prediction.round(12)
prediction.to_csv(prediction_path, index=False)      # Save prediction file.