import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define a custom dataset class
class H5Dataset(Dataset):
    def __init__(self, h5_file, data_key, label_key, transform=None):
        """
        Args:
            h5_file (str): Path to the .h5 file
            data_key (str): The key for the data (features) in the .h5 file
            label_key (str): The key for the labels in the .h5 file
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Open the .h5 file and load the data and labels
        with h5py.File(h5_file, 'r') as f:
            self.data = np.array(f[data_key])
            self.labels = np.array(f[label_key])
        
        # Convert to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Load your .h5 file
h5_file = 'dataset/fed_emnist_digitsonly_test.h5'
data_key = 'data'  # Replace with the correct key for your data
label_key = 'labels'  # Replace with the correct key for your labels

# Create dataset object
dataset = H5Dataset(h5_file=h5_file, data_key=data_key, label_key=label_key)

# Create DataLoader for batching
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Check the first batch (for demonstration purposes)
for batch in dataloader:
    print(batch['data'].shape, batch['label'].shape)
    break  # Only show the first batch

