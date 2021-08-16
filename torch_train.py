import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_model import SimpleFF
from util import split, split_dataframe, sample


def parse_df(df: pd.DataFrame, batch_size: int = 10) -> \
        ([np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    # print(df)
    # del df['Unnamed: 0']
    # df['age'] = pd.Series(df['age'].values.astype(np.float32))
    # df['gender'] = pd.Series(df['gender'].values.astype(np.float32))
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


def torch_train(df: pd.DataFrame):

    predicting = True

    epochs = 14
    batches_per_epoch = 4616
    input_len = 1
    nr_params = 12

    batch_size = (len(df) // batches_per_epoch)

    # specify columns to drop from dataframe for training
    to_del = ['Unnamed: 0', 'age', 'gender', 'rr_trend', 'rs_mean', 'rs_std', 'rr_range']
    df = df.drop(to_del, axis=1)
    nr_params -= len(to_del)

    data, train_in, train_out, test_in, test_out = parse_df(df, batch_size)

    model = SimpleFF((nr_params,))
    # model = CNN1D((input_len, nr_params))
    print(model)
    train_in = train_in.reshape(train_in.shape[0] // input_len, input_len, nr_params)
    # train_in = train_in.reshape(train_in.shape[0] // input_len, nr_params)
    train_out = np.asarray(train_out).reshape(train_out.shape[0] // input_len, 4)

    test_in = test_in.reshape(test_in.shape[0] // input_len, input_len, nr_params)
    # test_in = test_in.reshape(test_in.shape[0] // input_len, nr_params)
    test_out = np.asarray(test_out).reshape(test_out.shape[0] // input_len, 4)