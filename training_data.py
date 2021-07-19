import numpy as np
from matplotlib import pyplot as plt


# TODO: Write this class :)
class TrainingInterval:

    def __init__(self, resp_rates: np.ndarray, resp_strns: np.ndarray,
                 age: int, gender: int, label: str, prev: 'TrainingInterval'):
        self._resp_rates = resp_rates
        self._resp_strns = resp_strns
        self.age = age
        self.gender = gender
        self.label = label
        self.prev = prev

        self.mean_RR = np.mean(self._resp_rates)
        self.mean_RS = np.mean(self._resp_strns)
        self.trend_RR = None

        self._calc_stats()

    def _calc_stats(self):

        # the variance of our respiratory rates and strengths
        self.var_RR = np.var(self._resp_rates)
        self.var_RS = np.var(self._resp_strns)

        # standard deviation is the square root of variance
        self.std_RR = np.sqrt(self.var_RR)
        self.std_RS = np.sqrt(self.var_RS)

        # the range of values in our respiratory rates and strengths
        self.range_RR = np.max(self._resp_rates) - np.min(self._resp_rates)
        self.range_RS = np.max(self._resp_strns) - np.min(self._resp_strns)

        # the difference between the mean respiratory data in this interval and the previous
        if self.prev is not None:
            self.delta_RR = self.mean_RR - self.prev.mean_RR
            self.delta_RS = self.mean_RS - self.prev.mean_RS
        else:
            self.delta_RR = 0
            self.delta_RS = 0

    def export_stats(self) -> dict:

        return {
            "Respiratory Rate Mean": self.mean_RR,
            "Respiratory Strength Mean": self.mean_RS,
            "Respiratory Rate Standard Deviation": self.std_RR,
            "Respiratory Strength Standard Deviation": self.std_RS,
            "Respiratory Rate Variance": self.var_RR,
            "Respiratory Strength Variance": self.var_RS,
            "Respiratory Rate Range": self.range_RR,
            "Respiratory Strength Range": self.range_RS,
            "Respiratory Rate Mean Delta": self.delta_RR,
            "Respiratory Strength Mean Delta": self.delta_RS,
            "Gender": self.gender,
            "Age": self.age,
            "Label": self.label,
        }


# class for storing a collection of training intervals
class TrainingData:

    def __init__(self):
        self.intervals = []

    def _set_trend_values(self, trend_len: int = 5):
        rr_delta = [interval.delta_RR for interval in self.intervals]

        rr_delta_abs = np.abs(rr_delta)

        # the average change in rr over the past trend_len intervals
        for i, interval in enumerate(self.intervals):
            interval.trend_RR = np.mean(rr_delta_abs[i - trend_len:i]) if i > (trend_len - 1) else \
                np.mean(rr_delta_abs[0:i+trend_len])

        # self.trend_RR = [np.mean(rr_delta_abs[i - trend_len:i]) if i > (trend_len - 1) else 0.0 for i in
        #             range(len(rr_delta_abs))]

    def add(self, interval: TrainingInterval):
        self.intervals.append(interval)

    def trim(self):
        labels = np.array([interval.label for interval in self.intervals])
        light = np.where(labels == 'light')[0]
        deep = np.where(labels == 'deep')[0]
        rem = np.where(labels == 'rem')[0]

        # choose the intervals between our first and last asleep labels
        start = np.min([deep[0], light[0], rem[0]])
        end = np.min([deep[-1], light[-1], rem[-1]])
        self.intervals = self.intervals[start:end]

    def draw(self):
        xs = np.arange(len(self.intervals))
        self._set_trend_values(trend_len=10)

        rr_means = [interval.mean_RR for interval in self.intervals]
        rs_means = [interval.mean_RS for interval in self.intervals]

        rr_vars = [interval.var_RR for interval in self.intervals]
        rs_vars = [interval.var_RS for interval in self.intervals]

        rr_std = [interval.std_RR for interval in self.intervals]
        rs_std = [interval.std_RS for interval in self.intervals]
        rr_std_diff = [self.intervals[i].std_RR - self.intervals[i-1].std_RR for i in range(0, len(self.intervals))]

        rr_delta = [interval.delta_RR for interval in self.intervals]
        rs_delta = [interval.delta_RS for interval in self.intervals]

        rr_delta_abs = np.abs(rr_delta)
        rs_delta_abs = np.abs(rs_delta)

        rr_range = [interval.range_RR for interval in self.intervals]
        rs_range = [interval.range_RS for interval in self.intervals]

        rr_trend = [interval.trend_RR for interval in self.intervals]

        labels = [interval.label for interval in self.intervals]

        # plt.plot(xs, rr_means)
        plt.plot(xs, rs_means, c='g')
        # plt.plot(xs, rs_range, c='c')

        for i in range(len(xs)):
            if labels[i] == 'awake':
                plt.axvspan(i, i+1, facecolor='b', alpha=0.2)
            elif labels[i] == 'light':
                plt.axvspan(i, i+1, facecolor='b', alpha=0.4)
            elif labels[i] == 'deep':
                plt.axvspan(i, i+1, facecolor='b', alpha=0.6)
            elif labels[i] == 'rem':
                plt.axvspan(i, i+1, facecolor='r', alpha=0.6)

        plt.show()

    

