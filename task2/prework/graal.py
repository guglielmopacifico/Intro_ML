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

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier as OvRClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV

from tqdm.auto import tqdm

# ============================================================================

# VARIABLES

## Paths do data

train_feat = 'C:/Users/Lucas/OneDrive/Documents/Task2/train_features.csv'
train_output = 'C:/Users/Lucas/OneDrive/Documents/Task2/train_labels.csv'
prediction = 'C:/Users/Lucas/OneDrive/Documents/Task2/predictions.csv'
trans_train_feat = 'C:/Users/Lucas/OneDrive/Documents/Task2/train_features_transformed.csv'

test_feat = 'C:/Users/Lucas/OneDrive/Documents/Task2/test_features.csv'
trans_test_feat = 'C:/Users/Lucas/OneDrive/Documents/Task2/test_features_transformed.csv'

task1_out = 'C:/Users/Lucas/OneDrive/Documents/Task2/subtask1_output.csv'
task2_out = 'C:/Users/Lucas/OneDrive/Documents/Task2/subtask2_output.csv'
task3_out = 'C:/Users/Lucas/OneDrive/Documents/Task2/subtask3_output.csv'

## Sub-task 1 variables

eps = 10**-3   # Tolerance for stopping criteria
k_max = -1     # Max number of iterations
cache = 200    # Kernel cache size (in MB)
seed = 42      # Random seed for regression

## Labeling vectors

labels = ['pid','Time','Age','EtCO2','PTT','BUN','Lactate','Temp','Hgb','HCO3','BaseExcess','RRate','Fibrinogen','Phosphate','WBC','Creatinine','PaCO2','AST','FiO2','Platelets','SaO2','Glucose','ABPm','Magnesium','Potassium','ABPd','Calcium','Alkalinephos','SpO2','Bilirubin_direct','Chloride','Hct','Heartrate','Bilirubin_total','TroponinI','ABPs','pH']
new_labels = ['Age','EtCO2','PTT','BUN','Lactate','Temp','Hgb','HCO3','BaseExcess','RRate','Fibrinogen','Phosphate','WBC','Creatinine','PaCO2','AST','FiO2','Platelets','SaO2','Glucose','ABPm','Magnesium','Potassium','ABPd','Calcium','Alkalinephos','SpO2','Bilirubin_direct','Chloride','Hct','Heartrate','Bilirubin_total','TroponinI','ABPs','pH']
train_labels = ['LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2']
sepsis = ['LABEL_Sepsis']
regression = ['LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']

transform_feats = False

# ============================================================================

# TOOLS

def extract_data(path, patients):
    """Extracts data to be imputed from path."""

    data = pd.read_csv(path)

    data = data.to_numpy()[:12*patients] # Here only takes the first "patients" patients

    return data

def extract_labels(path, classifier=True, sepsis=False):
    """Extracts data from label file in path."""

    data = pd.read_csv(path)
    
    if classifier==True:
        if sepsis==False:
            y = data.to_numpy()[:,1:11] # Remove pid column and tasks 2,3 labels
        else:
            y = data.to_numpy()[:,11]
    else:
        y = data.to_numpy()[:,12:]
    
    return y

def get_features(data):
    """Transforms the features to a list, for all measurment types
    of the mean, median, min, max and number of measurements of that type;
    does imputation via the mean of the corresponding feature."""
    
    num_patient = int(data.shape[0]/12)     # Number of patients in sample
    num_mmt = int(data.shape[1])            # Number of measurements

    new_data = np.empty((num_patient, 5*num_mmt))

    for i in range(num_patient):                # Iterate over patients
        patient_data = data[i*12:(i+1)*12]      # Isolate patient i's data
        
        # # Initialise new data array for patient i
        # new_patient_data = np.array([])

        # Deal with the age
        age = patient_data[0,0]
        age_new_feat = [age, age, age, age, 12]
        new_data[i, :5] = np.array(age_new_feat)
        
        for j in range(1, num_mmt):             # Iterate over labels
            values = patient_data[:,j]          # Extract values of j-th patient

            # Compute new features
            mean = np.nanmean(values)
            median = np.nanmedian(values)
            mini = np.nanmin(values)
            maxi = np.nanmax(values)
            nans = int(np.sum(np.isnan(values)))

            # Save new features
            new_data[i, 5*j:5*(j+1)] = np.array([mean, median, mini, maxi, nans])
    
    means_all = np.nanmean(new_data, axis=0) # Means of each measurement used for imputation
    
    for i in range(num_patient):
        new_data[i][np.isnan(new_data[i])] = means_all[np.isnan(new_data[i])]
    
    return new_data

# ============================================================================

# DATA PRE-PROCESSING

if transform_feats==True:
    data = extract_data(train_feat,patients=18995) # Extraction of training data of first "patients" patients
    data = data[:,2:] # Remove pid and time stamps (indices 0 & 1)
    
    train_feat_trans = get_features(data) # Obtain the features we want
    pd.DataFrame(train_feat_trans).to_csv(trans_train_feat)
    
    data = extract_data(test_feat,patients=12664) # Extraction of test data of first "patients" patients
    data = data[:,2:] # Remove pid and time stamps (indices 0 & 1)
    
    test_feat_trans = get_features(data) # Obtain the features we want
    pd.DataFrame(test_feat_trans).to_csv(trans_test_feat)

print('Done with that')
# ================================================================================

# PRE-WORK

train_feat_trans = pd.read_csv(trans_train_feat).to_numpy()[:,1:] # Don't take first column
train_lbl_1 = extract_labels(train_output,classifier=True,sepsis=False)
train_lbl_2 = extract_labels(train_output,classifier=True,sepsis=True)
train_lbl_3 = extract_labels(train_output,classifier=False,sepsis=False)
test_feat_trans = pd.read_csv(trans_test_feat).to_numpy()[:,1:]

X_train = pd.DataFrame(train_feat_trans)
Y_train_1 = pd.DataFrame(train_lbl_1, columns=train_labels)
Y_train_2 = pd.DataFrame(train_lbl_2, columns=sepsis)
Y_train_3 = pd.DataFrame(train_lbl_3, columns=regression)

X_test = pd.DataFrame(test_feat_trans)

combined = pd.concat([X_train, Y_train_1],axis=1)
corr1 = combined.corr() # Matrix of correlations
corr1 = corr1.iloc[:X_train.shape[1],X_train.shape[1]:]

combined = pd.concat([X_train, Y_train_2],axis=1)
corr2 = combined.corr() # Matrix of correlations
corr2 = corr2.iloc[:X_train.shape[1],X_train.shape[1]:]

combined = pd.concat([X_train, Y_train_3],axis=1)
corr3 = combined.corr() # Matrix of correlations
corr3 = corr3.iloc[:X_train.shape[1],X_train.shape[1]:]

fig = plt.figure(figsize=(17.1, 1.5))
sns.heatmap(corr.T, square=True)
plt.show()

Y1 = pd.DataFrame(columns=train_labels)
Y2 = pd.DataFrame(columns=sepsis)
Y3 = pd.DataFrame(columns=regression)

n_feats = 30

for label in train_labels:
    
    ideal = corr1[label].nlargest(n=n_feats)
    ideal_index = ideal.index
    
    pipe = make_pipeline(StandardScaler(), SVC(random_state=seed, max_iter=50000, class_weight='balanced', probability=True))
    pipe.fit(X_train[ideal.index],Y_train_1[label])
    
    f_pred = pipe.decision_function(X_test[ideal_index])
    pred = 1/(1 + np.exp(-f_pred))
    print(pred)
    Y1[label] = pred
    #pred = pipe.predict_proba(X_test[ideal_index])[:,0]
    #Y1[label] = pred

print('Task1 okay')

ideal = corr2.nlargest(n=n_feats,columns='LABEL_Sepsis')
ideal_index = ideal.index

pipe = make_pipeline(StandardScaler(), SVC(random_state=seed, max_iter=50000, class_weight='balanced', probability=True))
pipe.fit(X_train[ideal.index],Y_train_2[sepsis])

f_pred = pipe.decision_function(X_test[ideal_index])
pred = 1/(1 + np.exp(-f_pred))
print(pred)
Y2['LABEL_Sepsis'] = pred

print('Task2 okay')

for label in regression:
    
    ideal = corr3[label].nlargest(n=n_feats)
    ideal_index = ideal.index
    
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
    pipe.fit(X_train[ideal.index],Y_train_3[label])
    
    pred = pipe.predict(X_test[ideal_index])
    Y3[label] = pred
    
print('Task3 okay')

Yte_df_1 = pd.DataFrame(Y1, columns=train_labels).to_csv(task1_out)
Yte_df_2 = pd.DataFrame(Y2, columns=sepsis).to_csv(task2_out)
Yte_df_3 = pd.DataFrame(Y3, columns=regression).to_csv(task3_out)
    
'''
corr = corr[train_labels].to_numpy()
corr_max = np.amax(np.abs(corr),axis=1)
max_indices = np.argsort(corr_max) # Get indices for sorted increasing correlations
max_indices = max_indices[-31:-1]
'''

Xtr = X_train.to_numpy()[:,max_indices] # This object has the training
                                         # data for all patients with the
                                         # 30 best correlated features
                                         # selected for the ML algorithms
Ytr = Y_train.to_numpy()

Xte = X_test.to_numpy()[:,max_indices]

'''
plt.figure(figsize=(36,30))
cor = X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
'''

# ================================================================================

# WORK

'''
# Initiate an SVM
svm = SVC(C=1.0,kernel='linear',tol=eps,cache_size=cache,random_state=seed,probability=True)
ovr = OvRClassifier(svm) # Multilabel classifier using svm for each label

# Fit data, obtain the ML model for subtask 1
ovr.fit(Xtr, Ytr)
Yte = ovr.predict_proba(Xte)
'''

## SUBTASK 1
# Make a pipeline to scale (via standard scaler) and with OvR estimator
#pipe = make_pipeline(StandardScaler(), OvRClassifier(SVC(probability=True)))
#pipe.fit(Xtr, Ytr)

## SUBTASK 2
# Make a pipeline to scale (via standard scaler) and with OvR estimator
#pipe = make_pipeline(StandardScaler(), SVC(probability=True))
#pipe.fit(Xtr, Ytr)

#Yte = pipe.predict_proba(Xte)
#Yte_df = pd.DataFrame(Yte, columns=train_labels).to_csv(task1_out)
#print(Yte_df)

# Predict the treatment outcome for testing data

'''
svm_mmt = measurments.main()    # Import subtask 1 fitted multilabel svm
svm_sepsis = sepsis.main()      # Import subtask 2 fitted svm

data = extract_data(test_feat,patients=1000) # extraction of data of first "patients" patients
pids = data[:,0][0::12]
data = data[:,2:] # Remove pid and time stamps BUT keep the pid somewhere
test_feat_trans = feat_transform(data)

imputer = KNNImputer(missing_values=np.nan,n_neighbors=k,weights='distance')
test_feat_imp = imputer.fit_transform(test_feat_trans)

svm_mmt = measurements.main()    # Import subtask 1 fitted multilabel svm
svm_sepsis = sepsis.main()      # Import subtask 2 fitted svm

predict.main()

# test_feat = measurments.feat_transform(test_feat)

# test_pred = svm_mmt.predict(test_feat)
# test_lbl = pd.read_csv(train_labels).to_numpy()[:, 1:11]

# diff = np.equal(test_lbl,test_pred)

# diff_df = pd.DataFrame(diff)

# diff_df.to_csv('task2_k49am2lqi/test.csv')
'''