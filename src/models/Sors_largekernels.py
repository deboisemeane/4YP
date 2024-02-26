import torch
import torch.nn as nn


# This implements a modified version of the model used by Sors et al. with larger kernels and fewer layers (no padding)
# It is designed to take feature length 3750 which corresponds to one 30s epoch at 125Hz, i.e. without surrounding context.
class Sors_largekernels(nn.Module):

    def __init__(self):
        super(Sors_largekernels, self).__init__()

        # Convolution layers
        padding23 = 11
        padding21 = 10
        padding19 = 9

        self.conv1 = nn.Conv1d(1, 128, 21, 2)
        self.conv2 = nn.Conv1d(128, 128, 21, 2)
        self.conv3 = nn.Conv1d(128, 128, 21, 2)
        self.conv4 = nn.Conv1d(128, 128, 21, 2)
        self.conv5 = nn.Conv1d(128, 128, 21, 2)
        self.conv6 = nn.Conv1d(128, 128, 21, 2)
        self.conv7 = nn.Conv1d(128, 128, 21, 2)

        self.fc1 = nn.Linear(1280, 100)
        self.fc2 = nn.Linear(100, 4)

        # Batch Normalisation
        self.norm0 = nn.BatchNorm1d(1)
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(128)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(128)
        self.norm5 = nn.BatchNorm1d(128)
        self.norm6 = nn.BatchNorm1d(128)
        self.norm7 = nn.BatchNorm1d(128)


    def forward(self, x):
        R = nn.LeakyReLU(0.1)
        x = R(self.norm1(self.conv1(self.norm0(x))))
        x = R(self.norm2(self.conv2(x)))
        x = R(self.norm3(self.conv3(x)))
        x = R(self.norm4(self.conv4(x)))
        x = R(self.norm5(self.conv5(x)))
        x = R(self.norm6(self.conv6(x)))
        x = R(self.norm7(self.conv7(x)))

        x = torch.flatten(x, start_dim=1)  # Start at dim1 such that the batch dimension is preserved.
        x = R(self.fc1(x))
        x = self.fc2(x)
        # Softmax is implemented in cross entropy loss, so we don't need it here.
        return x


if __name__ == '__main__':
    import numpy as np
    x_test = np.zeros((64, 1, 3750))
    x_test = torch.tensor(x_test, dtype=torch.float32)
    model = Sors_largekernels()
    print(model(x_test).shape)
