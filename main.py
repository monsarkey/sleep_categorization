from load_data import edf_to_csv
from util import split_dataframe, sample
import pandas as pd
# from keras.utils import to_categorical, np_utils
import numpy as np
from model import CNN1D

trimmed = False
trimmed_str = "_trimmed" if trimmed else ""

cleaned = True
cleaned_str = "_cleaned" if cleaned else ""

normalized = True
normalized_str = "_normalized" if cleaned else ""

epochs = 6
batch_size = 4
input_len = 1
nr_params = 10

if __name__ == '__main__':

    try:
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")
    except FileNotFoundError:
        print("file not found, reloading data from .edf")
        edf_to_csv(trimmed=trimmed, cleaned=cleaned, normalized=normalized)
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")

    # print(df)
    del df['Unnamed: 0']
    df['age'] = pd.Series(df['age'].values.astype(np.float32))
    df['gender'] = pd.Series(df['gender'].values.astype(np.float32))
    del df['age']
    batches = split_dataframe(df, batch_size=batch_size)
    data = [batch.values for batch in batches]

    train_data, test_data = sample(data, .9)
    del data

    # returns all columns but last, and then only last as np arrays
    def split(elt):
        return elt[:, :-1], elt[:, -1]

    train_in, train_out = map(list, zip(*[split(elt) for elt in train_data]))
    test_in, test_out = map(list, zip(*[split(elt) for elt in test_data]))

    train_in = np.concatenate(train_in)
    train_out = np.concatenate(train_out)
    test_in = np.concatenate(test_in)
    test_out = np.concatenate(test_out)

    # train_out = [pd.get_dummies(elt).values for elt in train_out]
    # test_out = [pd.get_dummies(elt).values for elt in test_out]

    train_out = pd.get_dummies(train_out)
    test_out = pd.get_dummies(test_out)

    # train_in = [np.asarray(elt).astype(np.float32) for elt in train_in]
    # test_in = [np.asarray(elt).astype(np.float32) for elt in test_in]

    train_in = np.asarray(train_in).astype(np.float32)
    test_in = np.asarray(test_in).astype(np.float32)

    # train_in.reshape(train_in.shape[0], )
    # train_out = np.asarray(train_out).astype(np.float32)



    # print(train_data)
    cnn = CNN1D((input_len, nr_params))
    train_in = train_in.reshape(train_in.shape[0]//input_len, input_len, nr_params)
    train_out = np.asarray(train_out).reshape(train_out.shape[0]//input_len, 4)

    test_in = test_in.reshape(test_in.shape[0]//input_len, input_len, nr_params)
    test_out = np.asarray(test_out).reshape(test_out.shape[0]//input_len, 4)
    cnn.fit(train_in, train_out, batch_size=batch_size, epochs=epochs)

    score = cnn.evaluate(test_in, test_out)
    print('accuracy: ', score[1] * 100, '%')

    # print(data)

    # print(df)