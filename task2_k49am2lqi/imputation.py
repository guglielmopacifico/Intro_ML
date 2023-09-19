# ============================================================================

# INTRODUCTION TO MACHINE LEARNING

# DATA IMPUTATION SCRIPT

# AUTHORS: M. Melennec, G. Pacifico, L. Tortora

# ============================================================================

# IMPORTS

import numpy as np

import pandas as pd

from sklearn.impute import KNNImputer

# ============================================================================

# TOOLS:

def extract_data(path):
    """Extracts data to be imputed from path."""

    data = pd.read_csv(path)

    data = data.to_numpy()[:]

    return data


# ============================================================================

# WORK

def main():

    import main

    data = extract_data(main.train_feat)       # Extaract data to impute

    # Impute values
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=main.k, 
                weights='distance')
    imp_data = imputer.fit_transform(data)

    return imp_data

if __name__ == "__main__":
    imp_data = main()