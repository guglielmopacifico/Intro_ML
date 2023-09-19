# ============================================================================

# INTRODUCTION TO MACHINE LEARNING

# TASK 4: Transfer Learning

# AUTHORS: M. Melennec, G. Pacifico, L. Tortora

# ============================================================================

# IMPORTS

import numpy as np
import pandas as pd
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import DataLoader, TensorDataset

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# ============================================================================

# VARIABLES

# Paths

dir = 'C:/Users/Lucas/OneDrive/Documents/Task4/' # Root directory

X_pretrain_path = dir + 'pretrain_features.csv'
y_pretrain_path = dir + 'pretrain_labels.csv'
X_train_path = dir + 'train_features.csv'
y_train_path = dir + 'train_labels.csv'
X_test_path = dir + 'test_features.csv'

save_path_test_enc = dir + 'train_features_encoded.csv'
save_path_predictions = dir + 'predictions.csv'

# Model variables

# Epochs
epoch_ae = 200
epoch_lumo = 100
epoch_predict = 100

bottleneck = 128            # Bottleneck size

seed = 0                   # Random seed

valid_prop = 0.20           # Proportion of data for validation

# Set device to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
else: torch.device("cpu")
print(f'Device used : {device}') 

torch.backends.cudnn.deterministic = True

# ============================================================================

# TOOLS AND FUNCTIONS

def evaluate(network, loader, loss_func):
    """
    Compute the test loss, to check for overfitting (lower is better)
    """
    network.eval()
    
    with torch.no_grad():
        valid_loss = 0.0
        
        for (X_valid, y_valid) in tqdm(loader):
            X_valid, y_valid = X_valid.to(device), y_valid.to(device)
            
            pred = network(X_valid)
            loss = loss_func(pred, y_valid)
            
            batch_size = X_valid.shape[0]
            valid_loss += loss * batch_size
                                  
        valid_loss /= len(loader)
    
    return valid_loss


def do_cool_stuff(network_name, network, loss_fn, epochs, train_loader,
                  valid_loader=None):
    """
    Training function (which does cool stuff) for a given network.
    """
    
    opti = torch.optim.Adam(network.parameters(), lr=1e-3)

    for epoch in range(1,epochs+1):

        train_loss = 0.0
        t = time.time()

        for (X_batch, y_batch) in tqdm(train_loader):
            opti.zero_grad()
            network.train()

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward
            pred = network(X_batch)
            loss = loss_fn(pred, y_batch)

            # Backward
            loss.backward()
            opti.step()

            batch_size = X_batch.shape[0]
            train_loss += loss * batch_size

        # average the accumulated statistics
        train_loss /= len(train_loader)
        if (valid_loader == None):
            test_loss = "None"
        else:
            test_loss = evaluate(network, valid_loader, loss_func).item()
        
        epoch_duration = time.time() - t
  
        # print statistics
        print(f'Epoch {epoch} | Train loss: {train_loss.item():.4f} | '
        f' Test loss: {test_loss:{"" if (valid_loader == None) else ".4f"}} |'
        f' Duration {epoch_duration:.2f} sec')

        # save checkpoint of network
        if epoch % 10 == 0 and epoch > 0:
            save_path = f'model_{network_name}_epoch_{epoch}.pt'
            torch.save({'epoch': epoch,
                        'model_state_dict': network.state_dict(),
                        'optimizer_state_dict': opti.state_dict()},
                       save_path)
            print(f'Saved model checkpoint to {save_path}')

# ============================================================================

# NEURAL NETWORKS

class AutoEncoder(torch.nn.Module):
    
    def __init__(self, num_features, bottleneck):
        super().__init__()

        acti = torch.nn.Tanh() # activation function

        self.encoding = nn.Sequential(Linear(num_features, 782), acti,
                                      Linear(782, 564), acti,
                                      Linear(564, 346), acti,
                                      Linear(346, bottleneck), acti,
                                      Linear(bottleneck, bottleneck))

        self.decoding = nn.Sequential(Linear(bottleneck, bottleneck), acti,
                                      Linear(bottleneck, 346), acti,
                                      Linear(346, 564), acti,
                                      Linear(564, 782), acti,
                                      Linear(782, num_features))

    def forward(self, X):
        X = self.encoding(X)
        X = self.decoding(X)
        return X
 
    def encode(self, X):
        X = self.encoding(X)
        return X


class Predictor(torch.nn.Module):
    
    def __init__(self, num_features, drop=0.2):
        super().__init__()
        
        #drop: Useful for dropping coefficients in second training (so as to reduce risk of overfitting on small set)
        acti = torch.nn.Tanh() # activation function
        dropout = torch.nn.Dropout(p=drop)
        dropout2 = torch.nn.Dropout(p=drop/2)

        self.predictor = nn.Sequential(dropout,
                                       Linear(num_features, 256), dropout, acti,
                                       Linear(256, 128), dropout, acti,
                                       Linear(128, 64), dropout2, acti,
                                       Linear(64, 32), acti, # From here don't need to dropout
                                       Linear(32, 16), acti,
                                       Linear(16, 1))
        
    def forward(self, X):
        X = self.predictor(X)
        return X

# ============================================================================

# WORK

# Data loading

pretrain_data = pd.read_csv(X_pretrain_path, index_col="Id").iloc[:,1:]
pretrain_labels = pd.read_csv(y_pretrain_path, index_col="Id")
train_data = pd.read_csv(X_train_path, index_col="Id").iloc[:,1:]
train_labels = pd.read_csv(y_train_path, index_col="Id")
test_data = pd.read_csv(X_test_path, index_col="Id").iloc[:,1:]

# Concatenate the pretraining, training and test dataset 
# to train the auto-encoder on all data
X_training = pd.concat([pretrain_data, train_data, test_data])

# Normalise the data
feature_scaler = preprocessing.StandardScaler()
ae_scaler = preprocessing.StandardScaler()

X_training_scaled = feature_scaler.fit_transform(X_training.values)
Xpretrain_rescaled = feature_scaler.transform(pretrain_data.values)
ypretrain_rescaled = ae_scaler.fit_transform(pretrain_labels.values)

# Split training and validation sets
X_train, X_valid, y_train, y_valid =\
    train_test_split(Xpretrain_rescaled, ypretrain_rescaled,
                     test_size=valid_prop, random_state=seed)
num_features = X_train.shape[1]
print('Input features:', num_features, '; Bottleneck features:', bottleneck)


# Pretraining

# Create data types correctly
train_ds = TensorDataset(torch.from_numpy(X_training_scaled).float(), 
                         torch.from_numpy(X_training_scaled).float())
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

# Do the training
network_ae = AutoEncoder(num_features, bottleneck).to(device)
loss_func = torch.nn.MSELoss()
do_cool_stuff('ae', network_ae, loss_func, epoch_ae, train_loader)

# Load epoch of trained autoencoder for further use
epoch_load = epoch_ae

model_ae_loaded = AutoEncoder(num_features, bottleneck)
checkpoint = torch.load(f'model_ae_epoch_{epoch_load}.pt')
model_ae_loaded.load_state_dict(checkpoint['model_state_dict'])
model_ae_loaded = model_ae_loaded.to(device)

model_ae_loaded.eval()


# Encode the training (or pre-training, i.e. LUMO) data using the auto-encoder that was just trained
with torch.no_grad():
    X_train_enc = model_ae_loaded.encode(torch.from_numpy(X_train).float().to(device))
    X_train_enc = X_train_enc.cpu()
    X_valid_enc = model_ae_loaded.encode(torch.from_numpy(X_valid).float().to(device))
    X_valid_enc = X_valid_enc.cpu()

# Create data types correctly
train_ds = TensorDataset(X_train_enc, torch.from_numpy(y_train).float()) # No need to modify y_train
valid_ds = TensorDataset(X_valid_enc, torch.from_numpy(y_valid).float()) # No need to modify y_train
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=512, shuffle=True)

# Do the training
predictor = Predictor(bottleneck).to(device)
loss_func = torch.nn.MSELoss()
do_cool_stuff('lumo', predictor, loss_func, epoch_lumo, train_loader, valid_loader)


# Transfer learning

# Get data and normalize it for gap
gap_scaler = preprocessing.StandardScaler()

X_train = feature_scaler.transform(train_data.values)
y_train = gap_scaler.fit_transform(train_labels.values)

# Encode the training (or real-training, i.e. HOMO-LUMO gap) data using the auto-encoder that was just trained
with torch.no_grad():
    X_train_enc = model_ae_loaded.encode(torch.from_numpy(X_train).float().to(device))
    X_train_enc = X_train_enc.cpu()

# Create data types correctly
train_ds = TensorDataset(X_train_enc, torch.from_numpy(y_train).float())
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

# Load specific version of pre-trained predictor to then train on gap
epoch_load = epoch_lumo

model = Predictor(bottleneck, 0.35)
checkpoint = torch.load(f'model_lumo_epoch_{epoch_load}.pt')
model.load_state_dict(checkpoint['model_state_dict'])
network = model.to(device)

# Do the training
loss_func = torch.nn.MSELoss()
do_cool_stuff('gap', network, loss_func, epoch_predict, train_loader)


# Predictions

# Load specific version of trained predictor to do the predictions on test set
epoch_load = epoch_predict

model_loaded = Predictor(bottleneck, 0.5)
checkpoint = torch.load(f'model_gap_epoch_{epoch_load}.pt')
model_loaded.load_state_dict(checkpoint['model_state_dict'])
model_loaded = model_loaded.to(device)

# Normalize the testing data and create column id's
X_test = feature_scaler.transform(test_data.values)
id_col = np.linspace(50100, 60099, 10000).astype(int)

# Predict the homo-lumo gap
model_loaded.eval()  
model_ae_loaded.eval()
with torch.no_grad():
    X_test_enc = model_loaded(model_ae_loaded.encode(torch.Tensor(X_test).cuda()))

y_test_predictions = gap_scaler.inverse_transform(X_test_enc.cpu().tolist())[:,0]

# Write it down somewhere nice
data = {'Id': id_col, 'y': y_test_predictions}
df_predictions = pd.DataFrame(data=data)
print(df_predictions)
df_predictions.to_csv(save_path_predictions,index=False)