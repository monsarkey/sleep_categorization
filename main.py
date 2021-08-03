from load_data import edf_to_csv
from util import split_dataframe, sample
import pandas as pd
# from keras.utils import to_categorical, np_utils
import tensorflow as tf
import numpy as np
from model import CNN1D, SimpleFF

trimmed = True
trimmed_str = "_trimmed" if trimmed else ""

cleaned = True
cleaned_str = "_cleaned" if cleaned else ""

normalized = True
normalized_str = "_normalized" if normalized else ""

predicting = True

epochs = 14
batches_per_epoch = 4616
input_len = 1
nr_params = 12

if __name__ == '__main__':

    try:
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")
    except FileNotFoundError:
        print("file not found, reloading data from .edf")
        edf_to_csv(trimmed=trimmed, cleaned=cleaned, normalized=normalized)
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")

    batch_size = (len(df) // batches_per_epoch)

    to_del = ['Unnamed: 0', 'age', 'gender', 'rr_trend', 'rs_mean', 'rs_std', 'rr_range']

    df = df.drop(to_del, axis=1)
    nr_params -= len(to_del)

    # print(df)
    # del df['Unnamed: 0']
    # df['age'] = pd.Series(df['age'].values.astype(np.float32))
    # df['gender'] = pd.Series(df['gender'].values.astype(np.float32))
    batches = split_dataframe(df, batch_size=batch_size)
    data = [batch.values for batch in batches]

    train_data, test_data = sample(data, .9)
    # del data

    # returns all columns but last, and then only last as np arrays
    def split(elt):
        return elt[:, :-1], elt[:, -1]

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

    # model = SimpleFF((nr_params,))
    model = CNN1D((input_len, nr_params))
    train_in = train_in.reshape(train_in.shape[0]//input_len, input_len, nr_params)
    # train_in = train_in.reshape(train_in.shape[0] // input_len, nr_params)
    train_out = np.asarray(train_out).reshape(train_out.shape[0]//input_len, 4)

    test_in = test_in.reshape(test_in.shape[0]//input_len, input_len, nr_params)
    # test_in = test_in.reshape(test_in.shape[0] // input_len, nr_params)
    test_out = np.asarray(test_out).reshape(test_out.shape[0]//input_len, 4)
    model.fit(train_in, train_out, batch_size=batch_size, epochs=epochs)

    score = model.evaluate(test_in, test_out)
    print('accuracy: ', score[1] * 100, '%')

    if predicting:
        predict_in, predict_out = map(list, zip(*[split(elt) for elt in data]))

        predict_in = np.concatenate(predict_in)
        predict_out = np.concatenate(predict_out)

        predict_in = np.asarray(predict_in).astype(np.float32)
        predict_out = pd.get_dummies(predict_out)

        predict_in = predict_in.reshape(predict_in.shape[0] // input_len, input_len, nr_params)
        # predict_in = predict_in.reshape(predict_in.shape[0] // input_len, nr_params)
        predict_out = np.asarray(predict_out).reshape(predict_out.shape[0] // input_len, 4)

        output = model.predict(predict_in, batch_size=batch_size)

        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(y_true=predict_out, y_pred=output)
        print(f"Categorical Accuracy after prediction (from keras): {m.result().numpy()}")
        # untrimmed weights
        # output[:, 2] -= .41
        # output[:, 1] += .22
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
        df.to_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}_output.csv")
        print('done!')
