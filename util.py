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
    num_batches = (len(df) // batch_size)
    for i in range(num_batches):
        batches.append(df[i * batch_size:(i+1) * batch_size])
    return batches


# returns all columns but last, and then only last as np arrays
def split(elt):
    return elt[:, :-1], elt[:, -1]


def parse_df(df: pd.DataFrame, batch_size: int = 10) -> \
        ([np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    batches = split_dataframe(df, batch_size=batch_size)
    data = [batch.values for batch in batches]

    train_data, test_data = sample(data, .9)

    train_in, train_out = map(list, zip(*[split(elt) for elt in train_data]))
    test_in, test_out = map(list, zip(*[split(elt) for elt in test_data]))

    train_in = np.concatenate(train_in)
    train_out = np.concatenate(train_out)
    test_in = np.concatenate(test_in)
    test_out = np.concatenate(test_out)

    train_out = pd.get_dummies(train_out)
    test_out = pd.get_dummies(test_out)

    train_in = np.asarray(train_in).astype(np.float32)
    test_in = np.asarray(test_in).astype(np.float32)

    return data, train_in, train_out, test_in, test_out
