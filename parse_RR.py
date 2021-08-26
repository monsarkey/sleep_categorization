import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from training_data import TrainingInterval, TrainingData
from util import standardize, normalize

"""
Author: Sean Markey (00smarkey@gmail.com)

Created: July 8th, 2021

This file contains the class definitions for DaySleep and RespiratoryEpoch, which are used in conjunction to parse
meaningful respiratory data from raw edf readings. This includes the parsing of ground truth labels from the
given .edf files, as well. 
"""


"""
The RespiratoryEpoch class contains a 30 second interval of inner-nasal voltage readings, from which we can find a
singular respiratory rate and strength reading. We do this by finding the time difference between peaks and valleys
in the voltage readings.
"""

class RespiratoryEpoch:

    def __init__(self, data: np.ndarray):
        self.data = data
        self.size = len(data)
        self._get_resp_rate()

    def _get_resp_rate(self, calc_str: bool = True):
        BR_STR_FACTOR = 1 / 50

        # identify peaks and valleys as relative extrema in voltage data
        self._peaks = argrelextrema(self.data, np.greater)[0]
        self._valleys = argrelextrema(self.data, np.less)[0]
        self._num_peaks = len(self._peaks)
        self._num_valleys = len(self._valleys)

        # I(n) = x(n + 1) - x(n) with n = 1, 2, 3, ..., N-1 where x(n) is the position of peak n in seconds
        self._breath_itvls = np.array([self._peaks[x + 1] - self._peaks[x] for x in range(self._num_peaks - 1)])

        # if we decide to calculate breathing strength, do it here:
        if calc_str:
            # we estimate the breathing strength as the average absolute voltage difference between peaks and valleys
            # in a respiratory epoch
            self._breath_delta = np.array(
                [self.data[self._peaks[x]] - self.data[self._valleys[x]] if x < (self._num_valleys - 1)
                 else 0 for x in range(self._num_peaks - 1)])

            if len(self._breath_delta) > 1:
                self.resp_strn = (BR_STR_FACTOR / len(self._breath_delta)) * np.sum(self._breath_delta)
            else:
                self.resp_strn = 0

        # respiratory rate = (60/N) * ( sum of 1 / I(n) from 1 to N)
        def resp_rate(N: int = 0) -> float:

            if N == 0:
                return 0.0
            sum = 0
            for i in range(N):
                sum += (1 / self._breath_itvls[i])
            return (60 / N) * sum

        self.resp_rate = resp_rate(self._num_peaks - 1)

    # draws a singular respiratory epoch with red dots labeling its peaks
    def draw(self):
        xs = np.arange(self.size)

        plt.plot(xs, self.data)
        plt.scatter(self._peaks, self.data[self._peaks], c='r')
        plt.show()

    # saves a singular respiratory epoch with red dots labeling its peaks
    def save_plot(self, filename: str = "figures/cur_epoch"):
        xs = np.arange(self.size)

        plt.plot(xs, self.data)
        plt.scatter(self._peaks, self.data[self._peaks], c='r')
        plt.savefig(filename)
        plt.close()

"""
The DaySleep class contains a list of 30s respiratory epochs, from which we obtain a list of respiratory rates
and strengths at every timestep. These are combined with a list of ground truth labels parsed from the .edf data,
tracking the sleep stages in 30s intervals. 
"""

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

    def __init__(self, day_num: int = 0, age: int = None, gender: int = None, data: np.ndarray = None,
                 labels: [float, bytes, str] = None):
        self.day_num = day_num
        self.age = age
        self.gender = gender
        self.data = data
        self.epochs = []
        self.labels = labels

        self.parse_labels(labels)
        self.split_epochs()
        self.resp_rates, self.resp_strns = self._get_resp()

    # use our list of respiratory epochs to get lists of respiratory rate and strength
    def _get_resp(self):
        return np.array([x.resp_rate for x in self.epochs]), \
               np.array([x.resp_strn for x in self.epochs])

    # draw our list of respiratory rates and strengths
    def draw_resp(self, filename: str, count: int = 0, normalized: bool = False,
                  standardized: bool = False, debug: bool = False):
        if len(self.epochs) == 0:
            return

        if len(self.resp_rates) == 0 or len(self.resp_strns) == 0:
            self.resp_rates, self.resp_strns = self._get_resp()

        debug_str = "/debug" if debug else ""

        # draw the standardized or normalized data depending on parameters
        xs = np.arange(len(self.resp_rates))
        if normalized or standardized:
            if normalized:
                norm_rates = normalize(self.resp_rates)
                norm_strns = normalize(self.resp_strns)

                plt.scatter(xs, norm_rates, s=1)
                plt.scatter(xs, norm_strns, s=1, c='g')
                plt.title(f"Normalized Breathing Rate / Str vs. 30s Epoch")
                plt.savefig(f"figures{debug_str}/days/day{count + 1}_normalized.png")
                plt.close()
            if standardized:
                std_rates = standardize(self.resp_rates)
                std_strns = standardize(self.resp_strns)

                plt.scatter(xs, std_rates, s=1)
                plt.scatter(xs, std_strns, s=1, c='g')
                plt.title(f"Standardized Breathing Rate / Str vs. 30s Epoch")
                plt.savefig(f"figures{debug_str}/days/day{count + 1}_standardized.png")
                plt.close()
        else:
            plt.scatter(xs, self.resp_rates, s=1)
            plt.scatter(xs, self.resp_strns, s=1, c='g')
            plt.title(f"Breathing Rate (breaths / min) and Str vs. 30s Epoch")
            plt.savefig(f"figures{debug_str}/days/day{count + 1}.png")

        plt.close()

    # here we do our best to clean the respiratory data
    def clean_data(self):
        if len(self.resp_rates) == 0 or len(self.resp_strns) == 0:
            self.resp_rates, self.resp_strns = self._get_resp()

        # find the standardized rates and strengths
        self.std_rates = standardize(self.resp_rates)
        self.std_strns = standardize(self.resp_strns)

        df = pd.DataFrame(data={
            'br. rates': self.resp_rates,
            'br. strns': self.resp_strns,
            'br. rates std.': self.std_rates,
            'br. strns std.': self.std_strns,
        })

        # find and eliminate all respiratory rates and strengths outside of a certain amount of standard deviations
        df_filtered = df[
            (np.abs(df['br. rates std.']) < 2.5) &
            (np.abs(df['br. strns std.']) < 3.5)
        ]

        # interpolate removed values
        df_filtered = df_filtered.reindex(index=(np.arange(len(df))), method='nearest')
        self.resp_rates = df_filtered['br. rates']
        self.resp_strns = df_filtered['br. strns']

    # separate timestamped sleep stages into 30s intervals
    def parse_labels(self, labels: [float, bytes, str], interval: int = 30):

        if labels is None:
            return

        label_arr = []

        for label in labels:
            for _ in range(int(label[1].decode('utf-8')) // interval):
                label_arr.append(DaySleep.stage_labels[label[2]])

        self.labels = label_arr

    # this function splits our data back into a collection of 30s intervals such that they line up with our
    # ground truth labels. We use these 30s collections of respiratory data for feature selection
    def get_intervals(self, interval: int = 30, trimmed: bool = False, normalized: bool = False) -> TrainingData:
        intervals_arr = []

        if self.data is None:
            return None

        # split our rates and strengths into 30 intervals
        sub_rates = np.array_split(self.resp_rates, len(self.resp_rates)//interval)
        sub_strns = np.array_split(self.resp_strns, len(self.resp_strns)//interval)

        prev_itvl = None
        for index in range(len(sub_rates)):
            # if we wanted, here we could pass in normalized or standardized resp_rates and resp_strns
            new_itvl = TrainingInterval(resp_rates=sub_rates[index], resp_strns=sub_strns[index],
                                        age=self.age, gender=self.gender, label=self.labels[index], prev=prev_itvl)
            intervals_arr.append(new_itvl)

            # pass in reference to previous training interval for analysis
            prev_itvl = new_itvl

        # initialize training data with this new array of intervals
        return TrainingData(intervals_arr, trimmed=trimmed, normalized=normalized)

    # here we split our voltage data into 30s resipiratory epochs, obtained as a 30s sliding window around every
    # second of data
    def split_epochs(self, interval: int = 30):

        if self.data is None:
            return

        # 86,400 seconds in a day for 86,400 data points
        self.data = np.resize(self.data, (86400,))

        # 30s window around every second in data is an epoch
        for index, point in enumerate(self.data):
            span = interval // 2
            size = len(self.data) - 1
            if index - span > 0 and index + span < size:
                cur_epoch = RespiratoryEpoch(data=self.data[(index - (interval // 2)):(index + (interval // 2))])
            elif index - span > 0:
                cur_epoch = RespiratoryEpoch(data=self.data[(index - (interval // 2)):size])
            elif index + span < size:
                cur_epoch = RespiratoryEpoch(data=self.data[0:(index + (interval // 2))])

            self.epochs.append(cur_epoch)
