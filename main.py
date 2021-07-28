from load_data import edf_to_csv
from util import split_dataframe, sample
import pandas as pd
from keras.utils import to_categorical, np_utils
import numpy as np
from model import CNN1D

trimmed = False
trimmed_str = "_trimmed" if trimmed else ""

cleaned = True
cleaned_str = "_cleaned" if cleaned else ""

normalized = True
normalized_str = "_normalized" if cleaned else ""

if __name__ == '__main__':

    try:
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")
    except FileNotFoundError:
        print("file not found, reloading data from .edf")
        edf_to_csv(trimmed=trimmed, cleaned=cleaned, normalized=normalized)
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")

    # print(df)
    del df['Unnamed: 0']
    batches = split_dataframe(df, batch_size=2880)
    data = [batch.values for batch in batches]

    train_data, test_data = sample(data, .9)
    del data

    # returns all columns but last, and then only last as np arrays
    def split(elt):
        return elt[:, :-1], elt[:, -1]

    train_in, train_out = map(list, zip(*[split(elt) for elt in train_data]))
    test_in, test_out = map(list, zip(*[split(elt) for elt in test_data]))

    train_out = [pd.Series(elt, dtype='category').values for elt in train_out]
    test_out = [pd.Series(elt, dtype='category').values for elt in test_out]
    print(train_data)
    # cnn = CNN1D((10, 1))
    # print(data)

    # print(df)