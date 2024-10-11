from torch.utils.data import DataLoader
from ..data.fire_dataset import FireDataset
import torch


dataset = FireDataset(inputs, targets)

batch_size = 32  # Adjust as needed
shuffle = True   # Set to False if you don't want to shuffle the data

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
