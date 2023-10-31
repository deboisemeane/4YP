import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):   # Multi-Layer Perceptron with either 1 or 2 hidden layers
    def __init__(self):
        super(MLP, self).__init__() # What do the arguments to super() mean here?
        self.fc1 = nn.Linear(20,10)
        self.fc2 = nn.Linear(10, 4)

        def forward(x):
            x = F.relu(self.fc1(x))
            x = F.softmax(self.fc2(x))
            return x






