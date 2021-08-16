import torch
import torch.nn as nn
import torch.nn.functional as F


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

        output = F.log_softmax(x, dim=1)
        return output
