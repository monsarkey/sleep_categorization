import numpy as np
import pandas as pd
import random


def standardize(arr: np.ndarray):
    if len(arr) > 0 and (std := np.std(arr)) != 0:
        return (arr - np.mean(arr)) / std
    else:
        return arr


def normalize(arr: np.ndarray):
    if (length := len(arr)) > 0 and (ptp := np.ptp(arr)) != 0:
        return (arr - np.min(arr)) / ptp
    elif length > 0:
        new_arr = np.empty(len(arr))
        return new_arr.fill(.5)
    else:
        return arr


def sample(lst: [np.ndarray], frac: float = 0.8) -> (list, list):
    indices = [elt[0] for elt in random.sample(list(enumerate(lst)), int(frac * len(lst)))]

    arr = np.array(lst, dtype=object)
    mask = np.ones(len(arr), dtype=bool)

    mask[indices] = False
    test = arr[mask]
    train = arr[~mask]

    return train, test


def split_dataframe(df: pd.DataFrame, batch_size: int = 2880) -> [pd.DataFrame]:
    batches = []
    num_batches = (len(df) // batch_size) + 1
    for i in range(num_batches):
        batches.append(df[i * batch_size:(i+1) * batch_size])
    return batches
