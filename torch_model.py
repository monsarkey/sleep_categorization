import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

"""
Author: Sean Markey (00smarkey@gmail.com)

Created: August 16th, 2021

This file holds the class definitions for all of my pytorch models.
"""

# this is a simple feed-forward model I used to get a baseline for how a model could work without time dependency
class SimpleFF(nn.Module):

    # just four fully connected layers
    def __init__(self, input_shape: tuple):
        super(SimpleFF, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], 20)
        self.fc2 = nn.Linear(20, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, 4)

    # apply ReLU after every fully connected layer, and then softmax on the outputs
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        output = F.softmax(x, dim=1)
        return output


# this is the start of an LSTM module, meant to take advantage of temporal dependencies in our 1D time series data
class LSTM(nn.Module):

    def __init__(self, input_size: int,
                 output_size: int = 4,
                 hidden_dim: int = 40,
                 num_layers: int = 1,
                 batch_size: int = 20,
                 seq_length: int = 20,
                 drop_prob: float = None):

        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.drop_prob = drop_prob
        self.batch_size = batch_size

        # if we have specified a drop probability, then add a dropout layer
        if drop_prob:
            self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, dropout=drop_prob, batch_first=True)
            self.dropout = nn.Dropout(drop_prob)
        else:
            self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)

        # two fully connected layers at the end to obtain output
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_size)

        self.relu = nn.ReLU()

    # define how data is processed through our model
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        out = hn.view(-1, self.hidden_dim)

        if self.drop_prob:
            out = self.dropout(out)
            out = self.relu(out)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)

        out = F.softmax(out, dim=1)

        return out

    def print(self):
        summary(self, input_size=(self.batch_size, self.seq_length, self.input_size))
