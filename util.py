import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt

from seaborn import heatmap
from sklearn.metrics import confusion_matrix

"""
Author: Sean Markey (00smarkey@gmail.com)

Created: July 19th, 2021

This file contains assorted utility functions used throughout the rest of the project. 
"""

# standardizes a numpy array, returning the array itself if it is empty or if its standard deviation is zero
def standardize(arr: np.ndarray):
    if len(arr) > 0 and (std := np.std(arr)) != 0:
        return (arr - np.mean(arr)) / std
    else:
        return arr

# normalizes a numpy array, returning an array filled with .5 if the range of the array is zero, and the array itself
# if it is empty
def normalize(arr: np.ndarray):
    if (length := len(arr)) > 0 and (ptp := np.ptp(arr)) != 0:
        return (arr - np.min(arr)) / ptp
    elif length > 0:
        new_arr = np.empty(len(arr))
        return new_arr.fill(.5)
    else:
        return arr

# takes in a list of numpy arrays, and samples a fraction of them for testing, keeping the rest for training
def sample(lst: [np.ndarray], frac: float = 0.8) -> (list, list):
    indices = [elt[0] for elt in random.sample(list(enumerate(lst)), int(frac * len(lst)))]

    arr = np.array(lst, dtype=object)
    mask = np.ones(len(arr), dtype=bool)

    mask[indices] = False
    test = arr[mask]
    train = arr[~mask]

    return train, test

# splits a dataframe into batches according to batch_size
def split_dataframe(df: pd.DataFrame, batch_size: int = 2880) -> [pd.DataFrame]:
    batches = []
    num_batches = (len(df) // batch_size)
    for i in range(num_batches):
        batches.append(df[i * batch_size:(i+1) * batch_size])
    return batches

# returns all columns but last, and then only last as np arrays
def split(elt):
    return elt[:, :-1], elt[:, -1]

# takes in a dataframe, which it then splits into numpy arrays for machine learning. This includes arrays for
# input and output of training and testing data.
def parse_df(df: pd.DataFrame, batch_size: int = 10, debug: bool = False) -> \
        ([np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    batches = split_dataframe(df, batch_size=batch_size)
    data = [batch.values for batch in batches]

    # if we are debugging, we will sample an equal amount of points from every label category
    if not debug:
        train_data, test_data = sample(data, .9)
    else:
        _, test_data = sample(data, .9)

        debug_sample_size = 5
        try:
            debug_train_ind = pd.read_csv(f"data/debug/set{debug_sample_size * 4}.csv").values.transpose()[1]
        except FileNotFoundError:

            debug_train_ind = []
            debug_train_ind.extend(
                df.loc[df['label'] == 'awake'].sample(debug_sample_size, random_state=1).index.values[:])
            debug_train_ind.extend(
                df.loc[df['label'] == 'light'].sample(debug_sample_size, random_state=1).index.values[:])
            debug_train_ind.extend(
                df.loc[df['label'] == 'deep'].sample(debug_sample_size, random_state=1).index.values[:])
            debug_train_ind.extend(
                df.loc[df['label'] == 'rem'].sample(debug_sample_size, random_state=1).index.values[:])
            pd.DataFrame(debug_train_ind).to_csv(f"data/debug/set{debug_sample_size * 4}.csv")

        train_data = df.iloc[debug_train_ind].values
        train_data = train_data.reshape(train_data.shape[0] // batch_size, batch_size, 6)
        # train_out = train_out.iloc[debug_train_ind]

    train_in, train_out = map(list, zip(*[split(elt) for elt in train_data]))
    test_in, test_out = map(list, zip(*[split(elt) for elt in test_data]))

    train_in = np.concatenate(train_in)
    train_out = np.concatenate(train_out)
    test_in = np.concatenate(test_in)
    test_out = np.concatenate(test_out)

    train_out = pd.get_dummies(train_out)[['awake', 'light', 'deep', 'rem']]
    test_out = pd.get_dummies(test_out)[['awake', 'light', 'deep', 'rem']]

    train_in = np.asarray(train_in).astype(np.float32)
    test_in = np.asarray(test_in).astype(np.float32)

    return data, train_in, train_out, test_in, test_out

# draw a confusion matrix according to a provided list of predicted values and ground truth values
def draw_conf(pred: [torch.Tensor], true: [torch.Tensor], name: str = None):
    columns = ['awake', 'light', 'deep', 'rem']

    if not isinstance(true, np.ndarray) or not isinstance(pred, np.ndarray):
        true = np.array([columns[val] for val in true])
        pred = np.array([columns[val] for val in pred])

    conf = confusion_matrix(true, pred, labels=columns)
    df_cm = pd.DataFrame(conf, index=columns, columns=columns)

    figure = plt.figure(figsize=(4, 4))
    heatmap(df_cm, annot=True, cmap="flare", fmt="d")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # if we have a name for this matrix, then save it, else just draw it.
    if name:
        plt.title(name)
        plt.savefig(f"figures/confusion_matrices/{name}")
        plt.close()
    else:
        plt.show()
