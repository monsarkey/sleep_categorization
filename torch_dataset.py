import numpy as np
from torch.utils.data import Dataset

"""
Author: Sean Markey (00smarkey@gmail.com)

Created: August 25th, 2021

This file holds the definition for my custom pytorch dataset. This allows me to have a sliding window of points around
each timestep for the purpose of training my LSTM. 
"""


class SlidingWindowDataset(Dataset):

    def __init__(self, input_arr: np.ndarray, label_arr: np.ndarray, window_size: int = 5, transform=None):
        self.input_arr = input_arr
        self.label_arr = label_arr
        self.window_size = window_size
        self.transform = transform
        self._mask, self._valid_indices = self._mark_windows()

    # mark the timesteps which have invalid windows (len < window_size or containing points from more than one night)
    def _mark_windows(self):

        mask_arr = np.ones(len(self.input_arr))

        for index, inputs in enumerate(self.input_arr):
            window = self.input_arr[max((index - self.window_size), 0):index].tolist()

            if len(window) < self.window_size:  # window is not long enough
                mask_arr[index] = 0
            elif len(set([elt[-1] for elt in window])) > 1:  # window contains points from more than one night
                mask_arr[index] = 0

        # remove the final day_num variable in the input array after use
        self.input_arr = self.input_arr[:, :-1]

        return mask_arr, np.nonzero(mask_arr)[0]

    def __len__(self):
        return len(self._valid_indices)

    # implementing function to get tuple of data by index
    def __getitem__(self, index) -> tuple:

        # use map to only take valid window labels and inputs
        index_map = self._valid_indices[index]

        label = self.label_arr[index_map - 1]
        inputs = self.input_arr[max((index_map - self.window_size), 0):index_map]

        return inputs, label
