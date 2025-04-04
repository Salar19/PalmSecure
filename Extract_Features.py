import torch
import numpy as np
import os
import pickle
import cv2 as cv
from models import MyDataset
from torch.utils.data import DataLoader
from models.ccnet import ccnet

import os

def getFileNames(txt_file_path):
    """Reads a text file containing image paths and extracts the last integer before the extension as labels."""
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    file_paths = []
    labels = []
    
    for line in lines:
        path = line.strip().split(' ')[0]  # Get the image path
        
        # Extract the last integer before the file extension as the label
        label = os.path.splitext(os.path.basename(path))[0]  # Remove extension from the file name
        label = label.split('/')[-1]  # Get the last part of the file path (last integer before .jpg)
        
        file_paths.append(path)
        labels.append(label)  # Use the extracted integer as the label

    return file_paths, labels

def extract_features_and_save(model, train_set_file, feature_save_path, batch_size=1):
    """
    Extract feature vectors for all images in the dataset and save them.

    Parameters:
    - model: The trained model
    - train_set_file: Path to the text file containing image paths for the training set
    - feature_save_path: Path to save the extracted feature vectors
    - batch_size: The batch size used for DataLoader
    """
    # Create DataLoader for the training set using MyDataset for the image loading
    trainset = MyDataset(txt=train_set_file, transforms=None, train=True, imside=128, outchannels=1)
    data_loader = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=12, shuffle=False)

    features = []
    labels = []

    # Iterate through the DataLoader
    for batch_id, (datas, target) in enumerate(data_loader):
        # Process the batch of images (each image is processed one at a time here)
        data = datas[0].cuda()  # Ensure image tensor is on GPU
        target = target.cuda()  # Ensure label tensor is on GPU

        # Extract feature for each image in the batch
        model.eval()
        with torch.no_grad():
            feature = model.getFeatureCode(data)  # Extract feature vector
            feature = feature.cpu().detach().numpy().flatten()  # Convert to numpy array

            # Append extracted features and corresponding labels
            features.append(feature)
            labels.append(target.cpu().detach().numpy())  # Convert label to numpy array

        print(f"Extracted feature for batch {batch_id + 1}/{len(data_loader)}")

    # Convert features and labels into numpy arrays
    features = np.array(features)
    labels = np.concatenate(labels)  # Concatenate all labels into one array

    # Ensure the save path exists
    if not os.path.exists(feature_save_path):
        os.makedirs(feature_save_path)

    # Save the features and labels as pickle files
    with open(os.path.join(feature_save_path, 'features.pkl'), 'wb') as f:
        pickle.dump(features, f)
    with open(os.path.join(feature_save_path, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)

    print(f"Feature vectors saved to {feature_save_path}")

# Example usage
if __name__ == "__main__":
    # Initialize the ccnet model
    model = ccnet(num_classes=767, weight=0.8)  # Modify the arguments if necessary
    model.load_state_dict(torch.load('newmodel.pth'))  # Load the pre-trained model
    model.cuda()  # Move the model to GPU if available
    
    # Path to the training set text file containing image paths
    train_set_file = "paths.txt"  # This is the text file with image paths
    feature_save_path = "./features"  # Path where the features will be saved

    # Extract features and save them
    extract_features_and_save(model, train_set_file, feature_save_path)
