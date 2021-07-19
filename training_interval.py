import numpy as np

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
        self.delta_RR = self.mean_RR - self.prev.mean_RR
        self.delta_RS = self.mean_RS - self.prev.mean_RS


