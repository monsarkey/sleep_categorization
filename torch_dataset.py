import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SlidingWindowDataset(Dataset):

    def __init__(self, input_arr: np.ndarray, label_arr: np.ndarray, window_size: int = 5, transform=None):
        self.input_arr = input_arr
        self.label_arr = label_arr
        self.window_size = window_size
        self.transform = transform
        self._mask, self._valid_indices = self._mark_windows()

    def _mark_windows(self):

        mask_arr = np.ones(len(self.input_arr))

        for index, inputs in enumerate(self.input_arr):
            window = self.input_arr[max((index - self.window_size), 0):index].tolist()

            if len(window) < self.window_size:
                mask_arr[index] = 0
            elif len(set([elt[-1] for elt in window])) > 1:
                mask_arr[index] = 0

        # remove the final day_num variable in the input array after use
        self.input_arr = self.input_arr[:, :-1]

        # drop indices identified earlier in mask
        # self.input_arr = self.input_arr[mask_arr == 1]
        # self.label_arr = self.label_arr[mask_arr == 1]

        return mask_arr, np.nonzero(mask_arr)[0]

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, index) -> tuple:

        index_map = self._valid_indices[index]

        label = self.label_arr[index_map-1]
        inputs = self.input_arr[max((index_map - self.window_size), 0):index_map]

        return inputs, label
