import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import median_absolute_deviation
from util import normalize


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

        self.disp_inv_RR = self.var_RR * self.mean_RR

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

    # def export_stats(self, normalized: bool = False) -> dict:
    #     # TODO: add new variables to export
    #     if normalized:
    #         return {
    #             "rr_mean": normalize(self.mean_RR),
    #             "rs_mean": normalize(self.mean_RS),
    #             "rr_std": normalize(self.std_RR),
    #             "rs_std": normalize(self.std_RS),
    #             "rr_range": normalize(self.range_RR),
    #             "rr_delta_abs": normalize(self.delta_abs_RR),
    #             "rs_delta_abs": normalize(self.delta_abs_RS),
    #             "rr_disp": 2 * normalize(self.disp_RR),
    #             "rr_trend": normalize(self.trend_RR),
    #             "gender": self.gender,
    #             "age": normalize(self.age),
    #             "label": self.label,
    #         }
    #     else:
    #         return {
    #             "rr_mean": self.mean_RR,
    #             "rs_mean": self.mean_RS,
    #             "rr_std": self.std_RR,
    #             "rs_std": self.std_RS,
    #             "rr_range": self.range_RR,
    #             "rr_delta_abs": self.delta_abs_RR,
    #             "rs_delta_abs": self.delta_abs_RS,
    #             "rr_disp": self.disp_RR,
    #             "rr_trend": self.trend_RR,
    #             "gender": self.gender,
    #             "age": self.age,
    #             "label": self.label,
    #         }


# class for storing a collection of training intervals
class TrainingData:

    count = 0

    def __init__(self, intervals: [TrainingInterval], trimmed: bool = False, normalized: bool = False):
        self.intervals = intervals
        self.trimmed = False
        self.normalized = normalized
        TrainingData.count += 1

        if trimmed:
            self.trim()

        self._set_trend_values(trend_len=10)

        self.rr_means = np.array([interval.mean_RR for interval in self.intervals])
        self.rs_means = np.array([interval.mean_RS for interval in self.intervals])

        self.rr_vars = np.array([interval.var_RR for interval in self.intervals])
        self.rs_vars = np.array([interval.var_RS for interval in self.intervals])

        self.rr_std = np.array([interval.std_RR for interval in self.intervals])
        self.rs_std = np.array([interval.std_RS for interval in self.intervals])

        self.rr_std_diff = np.array([self.intervals[i].std_RR - self.intervals[i-1].std_RR
                                     for i in range(0, len(self.intervals))])

        self.rr_delta = np.array([interval.delta_RR for interval in self.intervals])
        self.rs_delta = np.array([interval.delta_RS for interval in self.intervals])

        self.rr_delta_abs = np.array(np.abs(self.rr_delta))
        self.rs_delta_abs = np.array(np.abs(self.rs_delta))

        self.rr_range = np.array([interval.range_RR for interval in self.intervals])
        self.rs_range = np.array([interval.range_RS for interval in self.intervals])

        self.rr_trend = np.array([interval.trend_RR for interval in self.intervals])

        self.labels = np.array([interval.label for interval in self.intervals])

        self.rr_disp = np.array([interval.disp_RR for interval in self.intervals])
        self.rs_disp = np.array([interval.disp_RS for interval in self.intervals])

        self.rr_disp_inv = np.array([interval.disp_inv_RR for interval in self.intervals])

        self.gender = np.empty(len(self.intervals))
        self.gender.fill(self.intervals[0].gender)

    def _set_trend_values(self, trend_len: int = 5):
        rr_delta = [interval.delta_RR for interval in self.intervals]

        rr_delta_abs = np.abs(rr_delta)

        # the average change in rr over the past trend_len intervals
        for i, interval in enumerate(self.intervals):
            interval.trend_RR = np.mean(rr_delta_abs[i - trend_len:i]) if i > (trend_len - 1) else \
                np.mean(rr_delta_abs[0:i+trend_len])

        # self.trend_RR = [np.mean(rr_delta_abs[i - trend_len:i]) if i > (trend_len - 1) else 0.0 for i in
        #             range(len(rr_delta_abs))]

    def trim(self):
        if not self.trimmed:
            labels = np.array([interval.label for interval in self.intervals])
            light = np.where(labels == 'light')[0]
            deep = np.where(labels == 'deep')[0]
            rem = np.where(labels == 'rem')[0]
            # light = np.where(labels == 'nrem1')[0]
            # deep = np.where(labels == 'nrem3')[0]
            # rem = np.where(labels == 'rem')[0]

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

    def plot(self, draw_fig: bool = True, save_fig: bool = False, debug: bool = False):

        xs = np.arange(len(self.intervals))
        debug_str = "/debug" if debug else ""

        def draw_labels():
            for i in range(len(xs)):
                if self.labels[i] == 'awake':
                    plt.axvspan(i, i + 1, facecolor='b', alpha=0.2)
                elif self.labels[i] == 'light':
                    plt.axvspan(i, i + 1, facecolor='b', alpha=0.4)
                elif self.labels[i] == 'deep':
                    plt.axvspan(i, i + 1, facecolor='b', alpha=0.6)
                elif self.labels[i] == 'rem':
                    plt.axvspan(i, i + 1, facecolor='r', alpha=0.6)

        # plt.plot(xs, rr_means)
        if draw_fig:
            plt.plot(xs, self.rs_delta_abs, c='g', alpha=.7)
            draw_labels()
            plt.show()
        if save_fig:
            plt.plot(xs, self.rr_disp, c='g', alpha=.7)
            draw_labels()
            plt.title("Respiratory Rate Index of Dispersion vs. 30s Interval")
            plt.savefig(f"figures{debug_str}/vars/day{TrainingData.count}rr_disp")
            plt.close()
        # plt.plot(xs, rs_disp, c='c', alpha=.7)

    def export_stats(self) -> dict:
        # TODO: add new variables to export
        if self.normalized:
            return {
                "rr_mean": normalize(self.rr_means),
                "rs_mean": normalize(self.rs_means),
                "rr_std": normalize(self.rr_std),
                "rs_std": normalize(self.rs_std),
                "rr_range": normalize(self.rr_range),
                "rr_delta_abs": normalize(self.rr_delta_abs),
                "rs_delta_abs": normalize(self.rs_delta_abs),
                "rr_disp": 2 * normalize(self.rr_disp),
                "rr_trend": normalize(self.rr_trend),
                "gender": self.gender,
                "age": np.array([interval.age for interval in self.intervals]),
                "day_num": TrainingData.count,
                "label": self.labels,
            }
        else:
            return {
                "rr_mean": self.rr_means,
                "rs_mean": self.rs_means,
                "rr_std": self.rr_std,
                "rs_std": self.rs_std,
                "rr_range": self.rr_range,
                "rr_delta_abs": self.rr_delta_abs,
                "rs_delta_abs": self.rs_delta_abs,
                "rr_disp": self.rr_disp,
                "rr_trend": self.rr_trend,
                "gender": self.gender,
                "age": np.array([interval.age for interval in self.intervals]),
                "day_num": TrainingData.count,
                "label": self.labels,
            }

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.export_stats())

