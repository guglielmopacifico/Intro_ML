# ============================================================================

# INTRODUCTION TO MACHINE LEARNING

# TASK 3 - FOOD TASTE SIMILARITY PREDICTION

# AUTHORS: M. Melennec, G. Pacifico, L. Tortora
# GROUP: Zoomers

# ============================================================================

# IMPORTS

import random
import os
from tqdm import tqdm
import time

import numpy as np
from PIL import Image

import torch
from torch import nn#, cuda
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split

# ============================================================================

# VARIABLES

## User defined variables

# Paths
dir = 'task3/'

train_path = dir + 'train_triplets.txt'
test_path = dir + 'test_triplets.txt'
prediction_path = dir + 'predictions.txt'
trans_feat_root = dir + 'trans_food/'
model_save_path = dir + 'best_model.pt'

img_dir = dir + 'food/'

# Use pre-extracted image features
extract_feat = True

# Machine variables - device for computations
device = 'cpu'

## Internal variables

# Randomness variables
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# Pre-trained model variables
pt_model = models.efficientnet_b6(pretrained=True)
input_size = 528 # if use diff version of eff net, see input size at https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

# Own model variables
valid_prop = 0.10   # Proportion of train set to use for validation
triplet_loss = nn.TripletMarginLoss(margin=1.0,p=2)

in_size = 1280      # Size of transformed featrures

train_batch_size = 128
valid_batch_size = 128
test_batch_size = 128

num_epoch = 10

tolerance = 0.80

# ============================================================================

# TOOLS

def data_extract(path):
    """Extract data in path."""

    X = []

    with open(path, 'r') as file:
        # Extract content of each line
        for line in file:
            line = line.rstrip("\n")
            triplet = list(line.split(" "))
            
            # Save data
            X.append(triplet)
        
    return np.array(X)

def feat_extract(model, dataset, device):
    """Extract features of images from pretrained cnn without last layer."""

    # Set up pretrained model
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Identity(model.classifier[1].in_features)
    
    model.to(device)

    model.eval()

    # Extract features
    feats = dict()
    img_dl = DataLoader(dataset, batch_size=1, shuffle=False)
    print('Extracting features of images with pre-trained model')
    for id, img in tqdm(img_dl):
        id = id[0]
        img.to(device)
        img_feats = model(img)
        feats[id] = img_feats
    
    return feats

def predict(loader, network):
    """Predicts the loader values through network."""

    (A, B, C) = loader
    A_net, B_net, C_net = network(A, B, C)
    no_margin_loss = nn.TripletMarginLoss(margin=0.0, reduction='none')
    pred = torch.logical_not(no_margin_loss(A_net, B_net, C_net).data[0] > 0)

    return pred

def train(network, loss_func, train_loader, train_batch_size, valid_loader, 
        valid_truth, valid_batch_size, optim, epoch_max, tol, device, 
        save_path):
    """Train network."""

    network.to(device)

    # Iterate over epochs
    for epoch in range(epoch_max):

        # Reset epoch statistics
        train_loss = 0.0
        test_acc = 0.0
        t = time.time()

        for tid, (anchor, positive, negative) in \
                tqdm(enumerate(train_loader)):
            # Set device for loaders
            anchor.to(device)
            positive.to(device)
            negative.to(device)

            # Reset gradients and get into train mode
            optim.zero_grad()
            network.train()

            # Forward pass
            a_pred, p_pred, n_pred = network(anchor, positive, negative)
            loss = loss_func(a_pred, p_pred, n_pred)

            # Backward pass
            loss.backward()
            optim.step()

        # Upgrade epoch stats
            train_loss += loss.data / (train_batch_size*len(train_loader))
        
        network.eval()
        for valid_tid, valid_triplet in enumerate(valid_loader):
            pred = predict(valid_triplet, network)
            test_acc += (pred == valid_truth[valid_tid]) / (valid_batch_size*len(valid_loader))

        duration = time.time() - t

        # Print epoch stats
        print('Epoch {} | Train loss: {} | Test accuracy {} | Duration: {} sec'.format(epoch, train_loss, test_acc,duration))

        # Save first model
        if epoch == 0:
            min_acc = test_acc
            print('Saving model at ' + save_path)
            torch.save(network.state_dict(), save_path)
        
        # Save model if improvement
        if test_acc > min_acc:
            print('Test accuracy improvement. Saving model at ' + save_path)
            torch.save(network.sate_dict(), save_path)

        # Early stopping
        if test_acc > tol:
            print('Training finished: test loss reached threshold')
            return network
    
    return network

# ============================================================================

# CLASSES

class ImgDataset(Dataset):
    """A class to carry images and their ids in a Dataset"""

    def __init__(self, dir, transform, ext='.jpg'):
        super().__init__()

        self.img_dir = dir
        self.img_names = []
        for file in os.listdir(dir):
            if file.endswith(ext): self.img_names.append(file)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index, ext='.jpg'):
        name = self.img_names[index]
        path = self.img_dir + name
        id = name.strip(ext)
        pil_img = Image.open(path)
        trans_img = self.transform(pil_img)

        return id, trans_img

class TripletDataset(Dataset):
    """A class to carry triplets in a dataset"""

    def __init__(self, triplet_arr, feats):
        super().__init__()

        self.triplet_arr = triplet_arr
        self.features = feats
    
    def __len__(self):
        return self.triplet_arr.shape[0]

    def __getitem__(self, index):
        triplet = self.triplet_arr[index,:]
        triplet_feats = []
        for id in triplet:
            triplet_feats.append(self.features[id])

        return triplet_feats
        
class NeuralNet(nn.Module):
    """A class for the neural network"""

    def __init__(self, input_size, output_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                        kernel_size=(1,3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                        kernel_size=(1,3), stride=1)
        self.pool = nn.MaxPool2d(kernel_size=(1,3))
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(10, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.seq_conv = nn.Sequential(self.fc1, self.dropout, self.relu,
                            self.conv1, self.relu, self.pool,
                            self.conv2, self.relu, self.pool)
        self.seq_lin = nn.Sequential(self.fc2, self.dropout, self.relu,
                        self.fc3)

    def forward(self, anchor, positive, negative):
        anchor = self.seq_conv(anchor)
        anchor = self.dropout(torch.flatten(anchor, start_dim=1))
        anchor = self.seq_lin(anchor)

        positive = self.seq_conv(positive)
        positive = self.dropout(torch.flatten(positive, start_dim=1))
        positive = self.seq_lin(positive)

        negative = self.seq_conv(negative)
        negative = self.dropout(torch.flatten(negative, start_dim=1))
        negative = self.seq_lin(negative)
        return anchor, positive, negative




# ============================================================================

# PRE-WORK

# Define image transform
mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])

transform_images = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.Resize((input_size, input_size))])

# Transform images
if extract_feat:
    dataset = ImgDataset(img_dir, transform_images)
    feats = feat_extract(pt_model, dataset, device)

    # Save images
    if not os.path.isdir(trans_feat_root):
        os.mkdir(trans_feat_root)
    for id, features in feats.items():
        path = trans_feat_root + id + '.pt'
        torch.save(features, path)

else:
    feats = dict()
    for file in os.listdir(trans_feat_root):
        id = file.strip('.pt')
        feats[id] = torch.load(trans_feat_root+file)

print(feats['00000'].size())


# Extract trainig triplets
X_train = data_extract(train_path)
y_train = np.ones(X_train.shape[0])
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                        test_size=valid_prop,
                                        random_state=seed)

# Shuffle validation data
for i in range(len(y_valid)):
    y = random.randint(0,1)
    if not y:
        # Order B, C accordingly
        X_valid[i,1], X_valid[i, 2] = X_valid[i, 2], X_valid[i, 1]
        y_valid[i] = y

train_ds = TripletDataset(X_train, feats)
valid_ds = TripletDataset(X_valid, feats)

train_dl = DataLoader(train_ds)
valid_dl = DataLoader(valid_ds)

# Train network
network = NeuralNet(in_size, 24)
optim = torch.optim.Adam(network.parameters())

trained_net = train(network, triplet_loss, train_dl, train_batch_size,
                valid_dl, y_valid, valid_batch_size, optim, num_epoch,
                tolerance, device, model_save_path)

# Extract testing data
X_test = data_extract(test_path)
test_ds = TripletDataset(X_test, feats)
test_dl = DataLoader(test_ds)

test_pred = []
print('Making predictions...')
for loader in tqdm(test_dl):
    test_pred.append(predict(loader, network))

# Save predictions

with open(prediction_path, 'w') as file:
    for y in test_pred:
        file.write(str(y)+'\n')