# ============================================================================

# INTRODUCTION TO MACHINE LEARNING

# TASK2 - MEDICAL EVENTS PREDICTION
# CORRELATION OF VARIABLES

# AUTHORS: M. Melennec, G. Pacifico, L. Tortora

# ============================================================================

# IMPORTS

from matplotlib import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import seaborn as sns

# ============================================================================

# VARIABLES

num_pid = 10          # Number of random patients to pick

seed = 42               # Random seed

# Files containing data
train_feat = 'task2_k49am2lqi/train_features.csv'
train_labels = 'task2_k49am2lqi/train_labels.csv'

# ============================================================================

# TOOLS

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

    return new_data

# ============================================================================

# WORK
print(-2)
train_df = pd.read_csv(train_feat)              # Extract training data
print(-1)

# Random patients index selection
random.seed(seed)
rand_pid = random.sample(range(0,int(train_df.shape[0]/12)), num_pid)

train_df.drop(labels=['pid','Time'], axis=1)    # Remove useless columns
train_col = train_df.columns                    # save column names
# Select only the random k patients
timed_index = [range(i*12, (i+1)*12) for i in rand_pid]
train_df = train_df.iloc[timed_index]
print(0)

# Feature transform
train = train_df.to_numpy()
train_trans = feat_transform(train)
print(0.5)

# Change column names
transfos = ['mean', 'median', 'variance', 'min', 'max']
new_col = []
for col in train_col:
    for tr in transfos:
        new_col.append(tr+'_'+col)

train_df_trans = pd.DataFrame(train_trans, columns=new_col)


labels_df = pd.read_csv(train_labels)       # Extract labels
labels_df.drop(labels=['pid'], axis=1)      # Remove useless columns

# Select only the random k patients
labels_df = labels_df.iloc[[rand_pid]]
print(1)

# Fusion features and labels DataFrames
total = pd.concat([train_df_trans, labels_df], axis=1)

print(2)

# Get correlation data
plt.figure(figsize=(100,100))
cor = total.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.savefig('task2_k49am2lqi/correlation.pdf')
