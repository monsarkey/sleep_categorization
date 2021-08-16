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
    learning_rate = .0002
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
    # train_in = train_in.reshape(train_in.shape[0] // input_len, input_len, nr_params)
    train_in = train_in.reshape(train_in.shape[0] // input_len, nr_params)
    train_out = np.asarray(train_out).reshape(train_out.shape[0] // input_len, 4)
    # train_out = np.asarray(train_out).reshape(train_out.shape[0] // input_len)

    # test_in = test_in.reshape(test_in.shape[0] // input_len, input_len, nr_params)
    test_in = test_in.reshape(test_in.shape[0] // input_len, nr_params)
    test_out = np.asarray(test_out).reshape(test_out.shape[0] // input_len, 4)
    # test_out = np.asarray(test_out).reshape(test_out.shape[0] // input_len)

    train_data = torch.utils.data.TensorDataset(torch.tensor(train_in), torch.tensor(train_out))
    train_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = torch.utils.data.TensorDataset(torch.tensor(test_in), torch.tensor(test_out))
    test_load = torch.utils.data.DataLoader(test_data, batch_size=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_load)
    loss_list = []
    acc_list = []

    for epoch in range(epochs):
        print(f"Epoch #{epoch + 1}")
        batch_acc_list = []
        for i, (inputs, labels) in enumerate(train_load):

            outputs = model(inputs)
            labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            batch_acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{total_step}], "
                      f"Loss: {loss.item():.4f}, Accuracy: {((correct / total) * 100):.2f}%")

        print(f"Avg. Accuracy in Epoch #{epoch + 1}: {np.average(batch_acc_list) * 100:.2f}%")

    return df, model
