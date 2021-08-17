import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_model import SimpleFF

from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from util import parse_df


# def draw_acc(acc_arr: list, val_acc: float = None):
#     xs = np.arange(len(acc_arr))
#     plt.plot(xs, acc_arr)
#     if val_acc:
#         plt.axhline(y=val_acc, color='r')
#     plt.show()

def draw_epoch_acc(acc_arr: list, val_acc_arr: list):
    xs = np.arange(len(acc_arr))
    plt.plot(xs, acc_arr)
    plt.plot(xs, val_acc_arr, color='r')
    plt.show()

def draw_conf(pred, true, name: str = None):
    columns = ['awake', 'light', 'deep', 'rem']

    true = np.array([columns[val] for val in true])
    pred = np.array([columns[val] for val in pred])

    conf = confusion_matrix(true, pred, labels=columns)
    df_cm = pd.DataFrame(conf, index=columns, columns=columns)

    figure = plt.figure(figsize=(4, 4))
    heatmap(df_cm, annot=True, cmap="flare", fmt="d")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if name:
        plt.title(name)
        plt.savefig(f"figures/confusion_matrices/{name}")

    plt.show()

def modified_crossentropy_loss(pred, true):
    log_prob = -1.0 * F.log_softmax(pred, 1)
    loss = log_prob.gather(1, true.unsqueeze(1))
    loss = loss.mean()
    return loss


def torch_train(df: pd.DataFrame):

    predicting = True

    epochs = 1
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
    train_in = train_in.reshape(train_in.shape[0] // input_len, nr_params)
    train_out = np.asarray(train_out).reshape(train_out.shape[0] // input_len, 4)

    test_in = test_in.reshape(test_in.shape[0] // input_len, nr_params)
    test_out = np.asarray(test_out).reshape(test_out.shape[0] // input_len, 4)

    train_data = torch.utils.data.TensorDataset(torch.tensor(train_in), torch.tensor(train_out))
    train_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = torch.utils.data.TensorDataset(torch.tensor(test_in), torch.tensor(test_out))
    test_load = torch.utils.data.DataLoader(test_data, batch_size=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_load)
    loss_list = []
    epoch_acc_list = []
    epoch_val_acc_list = []

    for epoch in range(epochs):
        print(f"----------- Epoch #{epoch + 1} -----------")
        batch_acc_list = []
        val_batch_acc_list = []
        train_preds =[]
        train_vals = []
        val_preds = []
        val_labels = []
        for i, (inputs, labels) in enumerate(train_load):

            outputs = model(inputs)
            labels = torch.max(labels, 1)[1]
            # loss = criterion(outputs, labels)
            loss = modified_crossentropy_loss(outputs, labels)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            batch_acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{total_step}], "
                      f"Loss: {loss.item():.4f}, Accuracy: {((correct / total) * 100):.2f}%")

        epoch_acc = np.average(batch_acc_list)
        epoch_acc_list.append(epoch_acc)

        print(f"Avg. Accuracy in Training Epoch #{epoch + 1}: {np.average(batch_acc_list) * 100:.2f}%")

        for i, (inputs, labels) in enumerate(test_load):

            outputs = model(inputs)
            labels = torch.max(labels, 1)[1]

            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            val_batch_acc_list.append(correct / total)

        epoch_val_acc_list.append(np.average(val_batch_acc_list))
        print(f"Epoch #{epoch + 1} Validation Accuracy: {(np.average(val_batch_acc_list) * 100):.2f}%")


        # draw_acc(batch_acc_list)

    draw_epoch_acc(epoch_acc_list, epoch_val_acc_list)

    print("Testing: ")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        true = []
        pred = []
        for inputs, labels in test_load:
            outputs = model(inputs)
            labels = torch.max(labels, 1)[1]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            for i in range(len(labels)):
                true.append(labels[i].item())
                pred.append(predicted[i].item())

            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    print(f"Validation accuracy: {val_acc * 100:.2f}%")
    # draw_acc(epoch_acc_list, val_acc)
    draw_conf(pred, true, name="conf_final_validation")

    return df, model
