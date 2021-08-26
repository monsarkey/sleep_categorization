import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from math import exp
from torch_model import SimpleFF, LSTM
from torch_dataset import SlidingWindowDataset
from torch.optim.lr_scheduler import  StepLR

from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from util import parse_df, draw_conf


def draw_epoch_acc(acc_arr: list, val_acc_arr: list):
    xs = np.arange(len(acc_arr))
    plt.plot(xs, acc_arr)
    plt.plot(xs, val_acc_arr, color='r')
    plt.title("Accuracy per epoch in training and validation data")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy (%)")
    plt.show()

def draw_epoch_loss(loss_arr: list):
    xs = np.arange(len(loss_arr))
    plt.plot(xs, loss_arr)
    plt.title("Loss per epoch in training data")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()


def modified_crossentropy_loss(pred: [torch.Tensor], true: [torch.Tensor]) -> float:
    log_prob = -1.0 * F.log_softmax(pred, 1)

    loss = log_prob.gather(1, true.unsqueeze(1))
    for i in range(len(log_prob)):
        true_out = true[i]
        pred_out = pred[i].tolist().index(max(pred[i]))

        if pred_out != true_out and pred_out == 1:
            loss[i] *= 1.5

        if pred_out == 2 or pred_out == 3:
            loss[i] *= .8

    loss = loss.mean()
    return loss


def torch_train(df: pd.DataFrame):

    predicting = True
    debug = False

    epochs = 1
    learning_rate = .0001
    lr_step_size = 4
    lr_gamma = .5
    batches_per_epoch = 4000
    window_size = 10
    nr_params = len(df.columns) - 1

    if not debug:
        batch_size = (len(df) // batches_per_epoch)
    else:
        batch_size = 10

    # batch_size = 1

    # specify columns to drop from dataframe for training
    # to_del = ['Unnamed: 0', 'age', 'gender', 'rr_trend', 'rs_mean', 'rs_std', 'rr_range']
    to_del = ['Unnamed: 0', 'age', 'rr_std', 'rs_std', 'rr_disp']
    df = df.drop(to_del, axis=1)
    nr_params -= len(to_del)

    data, train_in, train_out, test_in, test_out = parse_df(df, batch_size, debug=debug)

    # model = SimpleFF((nr_params,))
    # train_in = train_in.reshape(train_in.shape[0] // input_len, nr_params)
    # train_out = np.asarray(train_out).reshape(train_out.shape[0] // input_len, 4)
    #
    # test_in = test_in.reshape(test_in.shape[0] // input_len, nr_params)
    # test_out = np.asarray(test_out).reshape(test_out.shape[0] // input_len, 4)

    # train_in = train_in[0:train_in.shape[0] - (train_in.shape[0] % seq_len)]
    # train_out = train_out[0:train_out.shape[0] - (train_out.shape[0] % seq_len)]
    #
    # train_in = train_in.reshape(train_in.shape[0] // seq_len, seq_len, nr_params)
    train_out = np.asarray(train_out).reshape(train_out.shape[0], 4)
    #
    # test_in = test_in[0:test_in.shape[0]-(test_in.shape[0] % seq_len)]
    # test_out = test_out[0:test_out.shape[0]-(test_out.shape[0] % seq_len)]
    #
    # test_in = test_in.reshape(test_in.shape[0] // seq_len, seq_len, nr_params)
    test_out = np.asarray(test_out).reshape(test_out.shape[0], 4)

    train_data = SlidingWindowDataset(train_in, train_out, window_size=window_size)
    # train_data = torch.utils.data.TensorDataset(torch.tensor(train_in), torch.tensor(train_out))
    train_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_data = torch.utils.data.TensorDataset(torch.tensor(test_in), torch.tensor(test_out))
    test_data = SlidingWindowDataset(test_in, test_out, window_size=window_size)
    test_load = torch.utils.data.DataLoader(test_data, batch_size=4)

    nr_params -= 1

    model = LSTM(input_size=nr_params, seq_length=window_size, num_layers=1, batch_size=batch_size, drop_prob=.3)

    # criterion = nn.CrossEntropyLoss(weight=torch.Tensor([2, 1.5, 5, 2]))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma, verbose=True)

    total_step = len(train_load)

    epoch_acc_list = []
    epoch_val_acc_list = []
    epoch_loss_list = []

    model.print()
    print(f"Training LSTM with sequence length = {window_size}")

    for epoch in range(epochs):
        print(f"----------- Epoch #{epoch + 1} -----------")

        # if epoch % 8 == 0:
        #     optimizer.param_groups[0]['lr'] *= .9

        batch_acc_list = []
        val_batch_acc_list = []
        train_preds = []
        train_labels = []
        val_preds = []
        val_labels = []

        loss_list = []

        for i, (inputs, labels) in enumerate(train_load):

            outputs = model(inputs)
            # outputs[:, 1] += .05
            # outputs = torch.Tensor([[elt / np.sum(probs.tolist()) for elt in probs] for probs in outputs], requires_grad=True)

            labels = torch.max(labels, 1)[1]
            # labels = torch.cat([torch.max(labels[num], 1)[1] for num in range(len(labels))])[0:len(outputs)]
            # loss = modified_crossentropy_loss(outputs, labels)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()

            train_preds.extend(predicted)
            train_labels.extend(labels)
            batch_acc_list.append(correct / total)

            # if (i + 1) % batch_size == 0:
            #     print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{total_step}], "
            #           f"Loss: {loss.item():.4f}, Accuracy: {((correct / total) * 100):.2f}%")

        epoch_acc = np.average(batch_acc_list)
        epoch_acc_list.append(epoch_acc)

        epoch_loss = np.average(loss_list)
        epoch_loss_list.append(epoch_loss)
        draw_conf(train_preds, train_labels, name=f"training/conf_epoch{epoch + 1}_train")

        print(f"Avg. Accuracy in Training Epoch #{epoch + 1}: {epoch_acc * 100:.2f}%")
        print(f"Avg. Loss in Training Epoch #{epoch + 1}: {epoch_loss:.4f}")

        for i, (inputs, labels) in enumerate(test_load):

            outputs = model(inputs)
            labels = torch.max(labels, 1)[1]
            # labels = torch.cat([torch.max(labels[num], 1)[1] for num in range(len(labels))])[0:len(outputs)]

            total = labels.size(0)
            # outputs[:, 1] -= .5
            # outputs[:, 0] -= .3
            # outputs[:, 3] += .3
            # outputs[:, 2] += .4
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()

            val_preds.extend(predicted)
            val_labels.extend(labels)
            val_batch_acc_list.append(correct / total)

        epoch_val_acc_list.append(np.average(val_batch_acc_list))
        draw_conf(val_preds, val_labels, name=f"validation/conf_epoch{epoch + 1}_validation")
        print(f"Epoch #{epoch + 1} Validation Accuracy: {(np.average(val_batch_acc_list) * 100):.2f}%")

        scheduler.step()


        # draw_acc(batch_acc_list)

    draw_epoch_acc(epoch_acc_list, epoch_val_acc_list)
    draw_epoch_loss(epoch_loss_list)

    print("----------- Testing ----------- ")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        true = []
        pred = []
        for inputs, labels in test_load:
            outputs = model(inputs)
            labels = torch.max(labels, 1)[1]
            # labels = torch.cat([torch.max(labels[num], 1)[1] for num in range(len(labels))])[0:len(outputs)]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            for i in range(len(labels)):
                true.append(labels[i].item())
                pred.append(predicted[i].item())

            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    # draw_acc(epoch_acc_list, val_acc)
    draw_conf(pred, true, name="conf_final_validation")

    return df, model
