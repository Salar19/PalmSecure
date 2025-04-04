import torch
import numpy as np
import pickle
import cv2 as cv
from models.ccnet import ccnet
import os
from torchvision import transforms as T
from PIL import Image

# NormSingleROI normalization class as provided
class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) with 0 mean and 1 std.
    [c,h,w]
    """
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size()

        if c != 1:
            raise TypeError('only support grayscale image.')

        tensor = tensor.view(c, h * w)
        idx = tensor > 0
        t = tensor[idx]

        m = t.mean()
        s = t.std() 
        t = t.sub_(m).div_(s + 1e-6)
        tensor[idx] = t
        
        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)
    
        return tensor

def preprocess_query_image(query_image_path, imside=128, outchannels=1):
    """
    Preprocess the query image in the same way as the images in the training set.
    """
    # Define the exact same transformations as in MyDataset
    transform = T.Compose([
        T.Resize((imside, imside)),  # Resize to the required dimensions
        T.ToTensor(),                # Convert to tensor
        NormSingleROI(outchannels)   # Normalize (using NormSingleROI)
    ])

    # Load the query image
    query_image = Image.open(query_image_path).convert('L')  # Open as grayscale image
    query_image = transform(query_image)  # Apply the transformations

    return query_image

def verify(model, query_image_path, feature_save_path, threshold=0.5, imside=128, outchannels=1):
    """
    Palmprint verification function.
    Compares a query image with images in the training set (database) and returns verification result.
    
    Parameters:
    - model: The trained model
    - query_image_path: Path to the query image
    - feature_save_path: Path where the features are stored
    - threshold: Similarity threshold for deciding if the query image matches a database image
    
    Returns:
    - True if a match is found (within threshold), False otherwise
    """
    
    # Step 1: Load pre-saved features and labels from the training set (database)
    with open(os.path.join(feature_save_path, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)
    
    with open(os.path.join(feature_save_path, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    
    print("Loaded saved features and labels.")

    # Step 2: Preprocess the query image using the same transformations as MyDataset
    query_image = preprocess_query_image(query_image_path, imside=imside, outchannels=outchannels)
    
    # Add batch dimension and move to GPU
    query_image = query_image.unsqueeze(0).cuda()

    # Step 3: Extract features for the query image using getFeatureCode (2048-dimensional)
    model.eval()
    with torch.no_grad():
        query_feature = model.getFeatureCode(query_image)  # Extract feature vector (2048-dimensional)
        query_feature = query_feature.cpu().detach().numpy().flatten()  # Convert to numpy array

    print(f"Query feature shape: {query_feature.shape}")
    print("Query image feature extracted.")
    
    # Step 4: Compare the query image's feature with each database image
    distances = []
    for i, db_feature in enumerate(features):
        # Cosine similarity between query and database image features
        cosdis = np.dot(query_feature, db_feature)  # Cosine similarity
        distances.append((cosdis, labels[i]))  # Store the similarity score and corresponding label (user ID)
    
    # Sort the distances to find the most similar image (smallest cosine distance)
    distances.sort(key=lambda x: x[0], reverse=True)  # Sort in descending order (highest similarity first)
    
    print(f"Query image compared with {len(distances)} database images.")
    
    # Step 5: Print similarity scores and labels for each comparison
    for i, (similarity, label) in enumerate(distances):
        print(f"Match {i+1}: Label = {label}, Similarity = {similarity:.4f}")
    
    # Step 6: Find the closest match (highest similarity)
    max_similarity, closest_user_id = distances[0]
    print(f"Closest match: User ID {closest_user_id} with similarity: {max_similarity:.4f}")
    
    # Step 7: Decision based on threshold
    if max_similarity > threshold:
        print(f"Verification successful. Query image matches with the database. Closest match label: {closest_user_id}")
        return True
    else:
        print("Verification failed. Query image does not match with the database.")
        return False


# Example usage
if __name__ == "__main__":
    # Initialize the ccnet model
    model = ccnet(num_classes=767, weight=0.8)  # Modify the arguments if necessary
    model.load_state_dict(torch.load('newmodel.pth'))  # Load the pre-trained model
    model.cuda()  # Move the model to GPU if available
    
    # Path to the query image
    query_image_path = "query.jpg"
    
    # Path to where the pre-saved feature vectors and labels are stored
    feature_save_path = "./features"
    
    # Perform the verification
    result = verify(model, query_image_path, feature_save_path=feature_save_path, threshold=0.5)
    
    if result:
        print("Palmprint verification successful!")
    else:
        print("Palmprint verification failed!")
