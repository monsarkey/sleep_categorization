import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SlidingWindowDataset(Dataset):

    def __init__(self, input_arr: np.ndarray, label_arr: np.ndarray, window_size: int = 5, transform=None):
        self.input_arr = input_arr
        self.label_arr = label_arr
        self.window_size = window_size
        self.transform = transform

    def __len__(self):
        return len(self.label_arr)

    def __getitem__(self, index) -> tuple:
        label = self.label_arr[index]
        inputs = self.input_arr[max((index - self.window_size), 0):index]

        return inputs, label
