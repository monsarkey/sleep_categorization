import pandas as pd
import numpy as np


class DaySleep:

    # stages nrem1 and nrem2 both "light" sleep, nrem3 and nrem4 "deep" sleep, awake, moving, and unk. all "awake"
    stage_labels = {
        'Sleep stage W': 'awake',
        'Movement time': 'awake',
        'Sleep stage ?': 'awake',
        'Sleep stage 1': 'light',
        'Sleep stage 2': 'light',
        'Sleep stage 3': 'deep',
        'Sleep stage 4': 'deep',
        'Sleep stage R': 'rem',
    }

    def __init__(self, age: int = None, gender: int = None, data: np.ndarray = None, labels: [float, bytes, str] = None):
        self.age = age
        self.gender = gender
        self.data = data
        self.breath_epochs = None
        self.rr_epochs = None
        self.labels = labels
        self.parse_labels(labels)
        self.split_epochs()

    # separate timestamped sleep stages into 30s intervals
    def parse_labels(self, labels: [float, bytes, str], interval: int = 30):

        if labels is None:
            return

        label_arr = []

        for label in labels:
            for _ in range(int(label[1].decode('utf-8')) // interval):
                label_arr.append(DaySleep.stage_labels[label[2]])

        self.labels = label_arr

    def split_epochs(self, interval: int = 30):

        if self.data is None:
            return

        # TODO: ensure that this resize is unnecessary (data is for some reason of different length than labels)
        self.data = np.resize(self.data, (86400,))
        self.breath_epochs = np.split(self.data, indices_or_sections=(self.data.size//interval))

#
# def get_rr(epoch: np.ndarray) -> float:
#     pass

