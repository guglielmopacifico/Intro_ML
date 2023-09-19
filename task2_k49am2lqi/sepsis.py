# ============================================================================

# INTRODUCTION TO MACHINE LEARNING

# TASK 2 - MEDICAL EVENTS PREDICTION
# SUB-TASK 2 - SEPSIS PREDICTION

# AUTHORS: M. Melennec, G. Pacifico, L. Tortora

# ============================================================================

# IMPORTS

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ============================================================================

# TOOLS

# ============================================================================

# WORK

# Relevent features
rel_feat = ['Temp', 'HCO3', 'Fibrinogen', 'Platelets', 'Glucose', 'ABPm',
            'Heartrate', 'Bilirubin_total']
# Relevent indices
rel_indices = [4, 6, 9, 16, 18, 19, 29, 30]

def main():

    import main

    train_feat = main.train_feat_trans
    train_lbl = main.train_lbl

    # Initiate an SVM
    svm = SVC(C=1.0, kernel='linear', tol=main.eps, cache_size=main.cache,
                random_state=main.seed)

    svm.fit(train_feat, train_lbl)

    return svm 

if __name__ == "__main__":
    main()