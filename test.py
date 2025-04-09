import tensorflow_datasets as tfds
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Load the EMNIST Digits dataset
dataset, info = tfds.load('emnist/digits', with_info=True, as_supervised=True)

# Get the train and test datasets
train_dataset, test_dataset = dataset['train'], dataset['test']

def tfds_to_pytorch(tfds_data):
    images = []
    labels = []
    for image, label in tfds_data:
        images.append(image.numpy())  # Convert TensorFlow tensor to NumPy array
        labels.append(label.numpy())  # Convert TensorFlow tensor to NumPy array
    # Convert to PyTorch tensors
    images_tensor = torch.tensor(np.array(images), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(labels), dtype=torch.long)
    return TensorDataset(images_tensor, labels_tensor)

# Convert to PyTorch Dataset
train_data = tfds_to_pytorch(train_dataset)
test_data = tfds_to_pytorch(test_dataset)

# Create DataLoader objects
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Example usage in a training loop
for images, labels in train_loader:
    print(images.shape, labels.shape)  # Use images and labels in your model

