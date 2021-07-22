import numpy as np


def standardize(arr: np.ndarray):
    if len(arr) > 0:
        return (arr - np.mean(arr)) / np.std(arr)
    else:
        return arr


def normalize(arr: np.ndarray):
    if len(arr) > 0 and (ptp := np.ptp(arr)) != 0:
        return (arr - np.min(arr)) / ptp
    elif len(arr > 0):
        new_arr = np.empty(len(arr))
        return new_arr.fill(.5)
    else:
        return arr
