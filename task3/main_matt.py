# ============================================================================

# INTRODUCTION TO MACHINE LEARNING

# TASK 3 - FOOD TASTE SIMILARITY PREDICTION

# AUTHORS: M. Melennec, G. Pacifico, L. Tortora
# GROUP: Zoomers

# ============================================================================

# IMPORTS

from pickletools import optimize
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import time

import math

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch import cuda

from keras import Input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16

from sklearn.model_selection import train_test_split

# ============================================================================

# VARIABLES

## Paths

dir = 'Task3/'                                  # Directory to data files

train_path = dir + 'train_triplets.txt'         # Training data file
test_path = dir + 'test_triplets.txt'           # Test data file
prediction_path = dir + 'predictions.txt'       # Predicition file
img_dir = dir + 'food/'                         # Directory to images
save_path = dir + 'best_model.pt'               # Path to best_model version in case of a crash

## Internal variables

# Images variables
ext = '.jpg'                # Extention for the image names

# Randomness variables
seed = 42                   # Seed for random generators
random.seed(seed)           # Set seed as default seed
torch.manual_seed(seed)

# Model variables
CNN_model = models.efficientnet_b0(pretrained=True)

pooling_method = 'avg'
resize = 244                  # Image input size
batch_size_train = 128
batch_size_valid = 128
batch_size_test = 128

# Loss function to be used for training
triplet_loss = nn.TripletMarginLoss(margin=1.0,p=2) # Want margin = 1, or margin = 0?

# ============================================================================

# TOOLS

## Data extraction

def data_extract(path, train=False):
    """Extract data in path. If train is set to true, the triplets will be 
    shuffled and the true order will also be returned."""

    X = []
    if train: y = []

    with open(path, 'r') as file:
        # Extract content of each line
        for line in file:
            line = line.rstrip("\n")
            triplet = list(line.split(" "))

            if train:
                # Shuffle data for training
                val = random.randint(0,1)
                if not val:
                    # Order B, C accordingly
                    triplet[1], triplet[2] = triplet[2], triplet[1]
            
            # Save data
            X.append(triplet)
            if train: y.append(val)

        if train:
            return np.array(X), np.array(y)
        
        return np.array(X)

## Data manipulation (transformation and image loading)

# New DataSet class (needed to overwrite __getitem__() function for this task)
class TripletImages(Dataset):
    
    def __init__(self, path, triplets, shuffle, loader=load_img, transform=None):
        """
        Parameters for builder
        ----------
        path : Path to food images
        triplets : Matrix of triplets such that triplets[i,j] returns the string for image on i-th row, j-th column
        shuffle : Vector of 1's and 0's returned from "data_extract" (0 if images have been shuffled, 1 otherwise)
        loader : Loader for the images (here using one from Keras)
        transform : Transformation to apply to images; set to None if already transformed

        Returns
        -------
        A DataSet instance of transformed images
        """        
        
        self.path = path
        self.triplets = triplets
        self.shuffle = shuffle
        self.transform = transform
        self.loader = loader
        
    def __getitem__(self, idx):
        """
        This is the important function to be changed to make our own version of a DataSet

        Parameters
        ----------
        idx : id over which the function iterates to extract all triplets

        Returns transformed triplets of images
        -------
        anchor : transformed anchor image
        positive : transformed positive image
        negative : transformed negative image
        """
        
        # Get all paths to the triplets images and check whether they have been shuffled
        img1, img2, img3 = self.triplets[idx,0], self.triplets[idx,1], self.triplets[idx,2]
        img1_path = self.path + img1 + ext
        img2_path = self.path + img2 + ext
        img3_path = self.path + img3 + ext
        shuffled = self.shuffle[idx]
        
        # Set anchor, positive and negative image for TripletLoss
        anchor = self.loader(img1_path)
        pairs = (self.loader(img2_path), self.loader(img3_path))
        positive, negative = pairs[shuffled^1], pairs[shuffled^0]
        
        """ MAYBE DELETE AT THE END
        if shuffled:
            positive, negative = self.loader(img2_path), self.loader(img3_path)
        else:
            positive, negative = self.loader(img3_path), self.loader(img2_path)
        """
            
        if self.transform!=None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            
        return anchor, positive, negative
        
    def __len__(self):
        """
        Since we redefine a new DataSet class, need to redefine this useful feature
        """
        return len(self.triplets)
        
def DataloaderImages(path, train_triplets, train_shuffle, valid_triplets, valid_shuffle, test_triplets, test_shuffle, batch_size_train, batch_size_valid, batch_size_test):
    """
    Function to return DataLoader of triplets used for training and testing (as well as validation), to be called in the main code after the "data_extract" function

    Parameters
    ----------
    path : Path to food images to feed to TripletImages class
    triplets : Matrix of triplets such that triplets[i,j] returns the string for image on i-th row, j-th column
    shuffle : Vector of 1's and 0's returned from "data_extract" (0 if images have been shuffled, 1 otherwise)
    /!\ In this case, we need to pass as arguments both the training and testing parameters, as well as validation set
    batch_size_train : Batch size for training triplets
    batch_size_valid : Batch size for validation triplets
    batch_size_test : Batch size for testing triplets

    Returns DataLoaders for training, validation and testing data
    -------
    loader_train : DataLoader of training data
    loader_valid : DataLoader of validation data
    loader_test : DataLoader of testing data
    """
    
    # Set-up the transform used for image pre-processing (use the same for both sets)
    mean, std = torch.Tensor([0.485, 0.456, 0.406]), torch.Tensor([0.229, 0.224, 0.225])
    
    transform_images = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize((resize, resize))])
    
    '''
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    '''
    
    trainset = TripletImages(path=path, triplets=train_triplets, shuffle=train_shuffle, transform=transform_images)
    loader_train = DataLoader(trainset, batch_size=batch_size_train)
    
    validset = TripletImages(path=path, triplets=valid_triplets, shuffle=valid_shuffle, transform=transform_images)
    loader_valid = DataLoader(validset, batch_size=batch_size_valid)
    
    testset = TripletImages(path=path, triplets=test_triplets, shuffle=test_shuffle, transform=transform_images)
    loader_test = DataLoader(testset, batch_size=batch_size_test)
    
    return loader_train, loader_valid, loader_test

## Neural network and machine learning algorithm

# Neural network model using "CNN_model" as pretrained initial NN and final embedding layer
# to use TripletLoss in training
class EmbeddedNN(nn.Module):
    
    def __init__(self, model):
        """
        Builder: needs an initial pretrained NN model

        Parameters
        ----------
        model : Pretrained model from NN PyTorch library; decided to use efficientnet_b0

        Returns
        -------
        A fully operational NN model with embedded final layer
        """
        
        super().__init__()
        # Want to extract out of pretrained model everything but last layer,
        # which we replace and sequentiate them into new
        model_CNN = model
        for param in self.model_CNN.parameters():
            param.requires_grad = False
        self.model_feats = nn.Sequential(model_CNN.children()[:-1])
        self.fc1 = nn.Linear(model.fc.in_features, 42)
        
    def forward(self, img):
        """
        Forward pass of our embedded network

        Parameters
        ----------
        img : Input image to pass through the network

        Returns
        -------
        Final output from 'concatenation' of pretrained model and final layer
        """
        
        out = self.model_feats(img)
        out = out.view(out.size(0), -1) # Correct shape for final layer
        out = self.fc1(out)
        
        return out

# CNN for food-taste learning; we use a pretrained model and an embedding as a final
# layer --> final CNN will consist of 3 embeddings with shared weights, akin to siamese
# network, such that for 3 input images we may create a comparison relation between them
class CNN_Food(nn.Module):
    
    def __init__(self, embedding):
        """
        Builder: uses NN model embedded with final layer manually (see above)
        """
        super().__init__()
        self.embedding = embedding
        
    def forward(self, anchor, positive, negative):
        """
        Forward pass of CNN: bring each image of triplet to embedded net
        """
        anchor_embedded = self.embedding(anchor)
        positive_embedded = self.embedding(positive)
        negative_embedded = self.embedding(negative)
        
        return anchor_embedded, positive_embedded, negative_embedded

def predict(loader, network):

    (A, B, C) = loader
    A_net, B_net, C_net = network(A, B, C)
    no_margin_loss = nn.TripletMarginLoss(margin=0.0, reduction='none')
    pred = no_margin_loss(A_net, B_net, C_net).data[0] == 0

    return pred

# Training function that runs through the images in different epochs and actually trains our
# model to evaluate taste similarity between images; as the name suggests, it actually does
# pretty cool stuff
def do_cool_stuff(network, train_loader, valid_loader, valid_truth, loss_func, opti, epoch_max, tol, train_batch_size, valid_batch_size, save_path):
    """
    Function that will train a neural network 

    Parameters
    ----------
    network : Neural Network to be trained
    loader_train : DataLoader of training set
    loader_valid : DataLoader of validation set
    loss : Loss to be used for training (TripletLoss in our case)
    optim : Optimizer used in gradient descent
    epoch_num : Number of training epochs we want
    gpu_is_ok : Whether a gpu is availableon your machine
    batch_size_train : see before
    batch_size_valid : see before

    Returns
    -------
    network : Trained network
    """

    # Check if a gpu is available
    if cuda.is_available:
        device = 'cuda:0'
    else: device = 'cpu'
    network.to(device)              # Set network to adequate device

    # Iterate over epochs
    for epoch in range(epoch_max):

        # Reset epoch statistics
        train_loss = 0.0
        acc = 0.0
        test_acc = 0.0
        t = time.time()

        # Iterate over triplets
        for tid, (a_loader, p_loader, n_loader) in enumerate(train_loader):
            # Set device for loaders
            a_loader.to(device)
            p_loader.to(device)
            n_loader.to(device)

            # Reset gradients and get into train mode
            opti.zero_grad()
            network.train()

            # Forward pass
            a_pred, p_pred, n_pred = network(a_loader, p_loader, n_loader)
            loss = loss_func(a_pred, p_pred, n_pred)

            # Backward pass
            loss.backward()
            opti.step()

        # Upgrade epoch stats
            train_loss += loss.data
            acc += train_loss>0
        avg_train_loss = train_loss / train_batch_size
        avg_acc = acc / train_batch_size
        network.eval()
        for valid_tid, valid_triplet in enumerate(valid_loader):
            pred = predict(valid_triplet)
            test_acc += pred == valid_truth[valid_tid]

        avg_test_acc = test_acc / valid_batch_size
        duration = time.time() - t

        # Print epoch stats
        print('Epoch {} | Train loss: {} | Train accuracy: {} | Test accuracy{} | Duration: {} sec'.format(epoch, avg_train_loss, avg_acc, avg_test_acc, duration))

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

# WORK

# Extract training data
Xtrain, ytrain = data_extract(train_path, train=True)

X_test = data_extract(test_path)
y_test = np.ones(X_test.shape[0]) # At beginning, naively set all 2nd column images to be the anchor

X_train, X_valid, y_train, y_valid = train_test_split(Xtrain, ytrain, test_size=0.23, random_state=seed)

train_loader, valid_loader, test_loader = DataloaderImages(img_dir, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size_train, batch_size_test, batch_size_valid)

"""
This code was copied from tutorial, and checks on first batch whether the data loaders are
correctly set up; will have to modify name of variables to the ones we use in this code
"""

# check that data loader is properly set up. We just check for the first batch
with torch.no_grad():
  for j, (x_batch, y_batch) in enumerate(train_loader):
    if j == 0:
      print('X batch shape', x_batch.shape)
      print('Y batch shape', y_batch.shape)
      print(x_batch[0,1].numpy())#.squeeze())
      s0=x_batch.shape
      s1=dataset[0][0].shape
      x_batch_img=torch.zeros((s0[0], s0[1], s1[0], s1[1], s1[2]), dtype=torch.float)
      for k in range(s0[0]):
        for l in range(s0[1]):
          x_batch_img[k,l,:,:,:]=dataset[x_batch[k,l]][0]
      print(x_batch_img.shape)
      #print(model.forward(x_batch_img) )
    if j > 0:
      break

'''
# Extract training data
X_test = data_extract(test_path)

y_pred = model.predict(X_test)                  # Make predictions

# Save predictions
with open(prediction_path, 'w') as pred:
    for y in y_pred:
        pred.write(str(int(y)) + '\n')
'''