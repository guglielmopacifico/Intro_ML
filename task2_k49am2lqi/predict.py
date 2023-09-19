# ============================================================================

# INTRODUCTION TO MACHINE LEARNING

# TASK 2 - MEDICAL EVENTS PREDICTION
# PREDICTION FILE

# AUTHORS: M. Melennec, G. Pacifico, L. Tortora

# ============================================================================

# IMPORTS

import numpy as np
import pandas as pd

import imputation
from sepsis import rel_feat

# ============================================================================

# TOOLS

# ============================================================================

# WORK

def main():
    
    import main

    test_feat = imputation.main()       # Extract and impute test features

    # Initialise prediction array
    test_pred = np.empty((trans_feat.shape[0], 16))
    test_pred[:,1] = test_feat[:, 1]        # Copy PIDs

    trans_feat = main.feat_transform(test_feat)     # Transform features

    # Predict with trained models
    test_pred[:, 1:11] = main.svm_mmt.predict_proba(trans_feat)
    test_pred[:, 11] = main.svm_sepsis.predict_proba(trans_feat[:,rel_feat])

    # Labels for the output file
    labels = ['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
                'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 
                'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis', 
                'LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
    # Build prediction DataFrame
    pred_df = pd.DataFrame(test_pred, columns=labels)

    pred_df.to_csv(main.prediction)

if __name__ == "__main__":
    main()