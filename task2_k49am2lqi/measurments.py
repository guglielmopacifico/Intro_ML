# ============================================================================

# INTRODUCTION TO MACHINE LEARNING

# TASK 2 - MEDICAL EVENTS PREDICTION
# SUB-TASK 1 - MEASURMENTS PREDICTION

# AUTHORS: M. Melennec, G. Pacifico, L. Tortora

# ============================================================================

# IMPORTS

import numpy as np

import pandas as pd
from pandas import DataFrame as df

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier as OvRClassifier

# ============================================================================

# TOOLS

# ============================================================================

# WORK

def main():

    import main

    ## Extract data
    train_feat = main.train_feat_trans
    train_lbl = main.train_lbl

    # Initiate an SVM
    svm = SVC(C=1.0, kernel='linear', tol=main.eps, cache_size=main.cache,
                random_state=main.seed)
    ovr = OvRClassifier(svm) # Multilabel classifier using svm for each label

    ## Fit data
    ovr.fit(train_feat, train_lbl)

    return ovr


if __name__ == '__main__':
    main()