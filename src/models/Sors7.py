import torch
import torch.nn as nn


# This implements a modified version of the model used by Sors et al.
# The final 5 convolution layers are replaced with a global max pooling layer.
class Sors7(nn.Module):

    def __init__(self):
        super(Sors7, self).__init__()

        # Convolution layers
        padding7 = 3
        padding5 = 2
        padding3 = 1

        self.conv1 = nn.Conv1d(1, 128, 7, 2, padding=padding7)
        self.conv2 = nn.Conv1d(128, 128, 7, 2, padding=padding7)
        self.conv3 = nn.Conv1d(128, 128, 7, 2, padding=padding7)
        self.conv4 = nn.Conv1d(128, 128, 7, 2, padding=padding5)
        self.conv5 = nn.Conv1d(128, 128, 7, 2, padding=padding5)
        self.conv6 = nn.Conv1d(128, 128, 7, 2, padding=padding7)
        self.conv7 = nn.Conv1d(128, 256, 7, 2, padding=padding7)

        # MLP
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 4)

        # Batch Normalisation
        self.norm0 = nn.BatchNorm1d(1)
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(128)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(128)
        self.norm5 = nn.BatchNorm1d(128)
        self.norm6 = nn.BatchNorm1d(128)
        self.norm7 = nn.BatchNorm1d(256)



    def forward(self, x):
        R = nn.LeakyReLU(0.1)
        x = R(self.norm1(self.conv1(self.norm0(x))))
        x = R(self.norm2(self.conv2(x)))
        x = R(self.norm3(self.conv3(x)))
        x = R(self.norm4(self.conv4(x)))
        x = R(self.norm5(self.conv5(x)))
        x = R(self.norm6(self.conv6(x)))
        x = R(self.norm7(self.conv7(x)))

        x = torch.max(x, 2)[0]
        x = R(self.fc1(x))
        x = self.fc2(x)
        # Softmax is implemented in cross entropy loss, so we don't need it here.
        return x


if __name__ == '__main__':
    import numpy as np
    x_test = np.zeros((64, 1, 15000))
    x_test = torch.tensor(x_test, dtype=torch.float32)
    model = Sors()
    print(model(x_test).shape)
