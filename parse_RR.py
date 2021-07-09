import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


class RespiratoryEpoch:

    def __init__(self, data: np.ndarray, size: int = 30):
        self.data = data
        self.size = len(data)
        self._get_resp_rate()

    def _get_resp_rate(self):
        self._peaks, _ = find_peaks(self.data, height=0)
        self._num_peaks = len(self._peaks)

    def draw(self):
        xs = np.arange(self.size)

        plt.plot(xs, self.data)
        plt.scatter(self._peaks, self.data[self._peaks], c='r')
        plt.show()


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
        self.epochs = []
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
        # split = np.split(self.data, indices_or_sections=(self.data.size//interval))

        # 30s window around every second in data is an epoch
        for index, point in enumerate(self.data):
            span = interval//2
            size = len(self.data) - 1
            if index - span > 0 and index + span < size:
                cur_epoch = RespiratoryEpoch(data=self.data[(index - (interval // 2)):(index + (interval // 2))])
            elif index - span > 0:
                cur_epoch = RespiratoryEpoch(data=self.data[(index - (interval // 2)):size])
            elif index + span < size:
                cur_epoch = RespiratoryEpoch(data=self.data[0:(index + (interval // 2))])

            cur_epoch.draw()
            self.epochs.append(cur_epoch)
