import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SimpleFF(nn.Module):

    def __init__(self, input_shape: tuple):
        super(SimpleFF, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], 20)
        self.fc2 = nn.Linear(20, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, 4)

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


class LSTM(nn.Module):

    def __init__(self, input_shape: tuple,
                 output_size: int = 4,
                 hidden_dim: int = 40,
                 num_layers: int = 1,
                 drop_prob: float = .25):

        super(LSTM. self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_shape[2], hidden_dim, num_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_size)

        self.relu = nn.reLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))

        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)
        out = self.dropout(hn)
        out = self.relu(out)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        return out
