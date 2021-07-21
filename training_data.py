import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import median_absolute_deviation


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

        self.median_RR = np.median(self._resp_rates)
        self.trend_RR = None

        self._calc_stats()

    def _calc_stats(self):

        # the variance of our respiratory rates and strengths
        self.var_RR = np.var(self._resp_rates)
        self.var_RS = np.var(self._resp_strns)

        # standard deviation is the square root of variance
        self.std_RR = np.sqrt(self.var_RR)
        self.std_RS = np.sqrt(self.var_RS)

        # self.var_coeff_RR =  10 * (1 + (1 / (4 * len(self._resp_rates) + 1))) * (self.std_RR / self.mean_RR)
        self.mad_RR = median_absolute_deviation(self._resp_rates)

        if self.mean_RR != 0:
            self.disp_RR = self.var_RR / self.mean_RR
        else:
            self.disp_RR = 0

        if self.mean_RS != 0:
            self.disp_RS = self.var_RS / self.mean_RS
        else:
            self.disp_RS = 0

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

        self.delta_abs_RR = np.abs(self.delta_RR)
        self.delta_abs_RS = np.abs(self.delta_RS)

    def export_stats(self) -> dict:
        # TODO: add new variables to export
        return {
            "rr_mean": self.mean_RR,
            "rs_mean": self.mean_RS,
            "rr_std": self.std_RR,
            "rs_std": self.std_RS,
            "rr_var": self.var_RR,
            "rs_var": self.var_RS,
            "rr_range": self.range_RR,
            "rs_range": self.range_RS,
            "rr_delta": self.delta_RR,
            "rs_delta": self.delta_RS,
            "rr_delta_abs": self.delta_abs_RR,
            "rs_delta_abs": self.delta_abs_RS,
            "rr_trend": self.trend_RR,
            "gender": self.gender,
            "age": self.age,
            "label": self.label,
        }


# class for storing a collection of training intervals
class TrainingData:

    count = 0

    def __init__(self):
        self.intervals = []
        self.trimmed = False
        TrainingData.count += 1

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
        try:
            start = np.min([deep[0], light[0], rem[0]])
            end = np.max([deep[-1], light[-1], rem[-1]])
        except IndexError:
            start = np.min([deep[0] if len(deep) > 0 else float('inf'),
                            light[0] if len(light) > 0 else float('inf'),
                            rem[0] if len(rem) > 0 else float('inf')])
            end = np.max([deep[-1] if len(deep) > 0 else float('-inf'),
                          light[-1] if len(light) > 0 else float('-inf'),
                          rem[-1] if len(rem) > 0 else float('-inf')])

        try:
            self.intervals = self.intervals[start:end]
        except TypeError:
            self.intervals = self.intervals
        self.trimmed = True

    def plot(self, draw_fig: bool = True, save_fig: bool = False):
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

        # testing area
        rr_median = [interval.median_RR for interval in self.intervals]
        rr_mad = [interval.mad_RR for interval in self.intervals]
        # rr_var_coeff = [interval.var_coeff_RR for interval in self.intervals]
        rr_disp = [interval.disp_RR for interval in self.intervals]
        rs_disp = [interval.disp_RS for interval in self.intervals]

        def draw_labels():
            for i in range(len(xs)):
                if labels[i] == 'awake':
                    plt.axvspan(i, i + 1, facecolor='b', alpha=0.2)
                elif labels[i] == 'light':
                    plt.axvspan(i, i + 1, facecolor='b', alpha=0.4)
                elif labels[i] == 'deep':
                    plt.axvspan(i, i + 1, facecolor='b', alpha=0.6)
                elif labels[i] == 'rem':
                    plt.axvspan(i, i + 1, facecolor='r', alpha=0.6)

        # plt.plot(xs, rr_means)
        if draw_fig:
            plt.plot(xs, rr_disp, c='g', alpha=.7)
            draw_labels()
            plt.show()
        if save_fig:
            plt.plot(xs, rr_disp, c='g', alpha=.7)
            draw_labels()
            plt.title("Respiratory Rate Index of Dispersion vs. 30s Interval")
            plt.savefig(f"figures/vars/day{TrainingData.count}rr_disp")
            plt.close()
        # plt.plot(xs, rs_disp, c='c', alpha=.7)

    def to_df(self) -> pd.DataFrame:
        self._set_trend_values(trend_len=10)
        dicts = [interval.export_stats() for interval in self.intervals]

        return pd.DataFrame(dicts)

