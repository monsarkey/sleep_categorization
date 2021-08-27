import pandas as pd
import tensorflow as tf
import numpy as np

from keras_model import CNN1D, SimpleFF
from util import parse_df, split

"""
Author: Sean Markey (00smarkey@gmail.com)

Created: July 27th, 2021

This file handles the cleaning of training data for the purpose of feeding into a keras model, along with the analysis
of this model after training
"""


def keras_train(df: pd.DataFrame):

    predicting = True

    # define hyperparameters of model
    epochs = 14
    batches_per_epoch = 4616
    input_len = 1
    nr_params = len(df.columns) - 1

    batch_size = (len(df) // batches_per_epoch)

    # specify columns to drop from dataframe for training
    to_del = ['Unnamed: 0', 'age', 'gender', 'rr_trend', 'rs_mean', 'rs_std', 'rr_range']
    df = df.drop(to_del, axis=1)
    nr_params -= len(to_del)

    data, train_in, train_out, test_in, test_out = parse_df(df, batch_size)

    # define model
    # model = SimpleFF((nr_params,))
    model = CNN1D((input_len, nr_params))

    # do reshaping on training and testing data
    train_in = train_in.reshape(train_in.shape[0] // input_len, input_len, nr_params)
    train_out = np.asarray(train_out).reshape(train_out.shape[0] // input_len, 4)

    test_in = test_in.reshape(test_in.shape[0] // input_len, input_len, nr_params)
    test_out = np.asarray(test_out).reshape(test_out.shape[0] // input_len, 4)

    # fit our model with this data
    model.fit(train_in, train_out, batch_size=batch_size, epochs=epochs)

    score = model.evaluate(test_in, test_out)
    print('accuracy: ', score[1] * 100, '%')

    # if we want, run this model on validation data to find accuracy reading
    if predicting:
        predict_in, predict_out = map(list, zip(*[split(elt) for elt in data]))

        predict_in = np.concatenate(predict_in)
        predict_out = np.concatenate(predict_out)

        predict_in = np.asarray(predict_in).astype(np.float32)
        predict_out = pd.get_dummies(predict_out)

        predict_in = predict_in.reshape(predict_in.shape[0] // input_len, input_len, nr_params)
        predict_out = np.asarray(predict_out).reshape(predict_out.shape[0] // input_len, 4)

        output = model.predict(predict_in, batch_size=batch_size)

        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(y_true=predict_out, y_pred=output)
        print(f"Categorical Accuracy after prediction (from keras): {m.result().numpy()}")

        # manually adjust weights of outputs (undesirable)
        output[:, 2] -= .5
        output[:, 1] += .3
        output[:, 3] -= .15

        predictions = np.array([elt.tolist().index(max(elt)) for elt in output])
        labels = np.array([['awake', 'light', 'deep', 'rem'][val] for val in predictions])
        df['out_label'] = labels

        incorr = np.where((df['label'] != df['out_label']), 1, 0)
        total_wrong = np.sum(incorr)
        print(f"Categorical Accuracy after prediction (manually computed): {1 - (total_wrong / len(predictions))}")
        df['incorr'] = incorr

        df = df.join(pd.DataFrame(output, columns=['awake%', 'light%', 'deep%', 'rem%']))

        return df, model

    return None, model
