import torch
import torch.nn as nn


# This implements the model used by Sors et al.
# It is designed to take feature length 1500, otherwise the fully connected layers will not work.
class Sors(nn.Module):

    def __init__(self):
        super(Sors, self).__init__()
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
        self.conv8 = nn.Conv1d(256, 256, 5, 2, padding=padding5)
        self.conv9 = nn.Conv1d(256, 256, 5, 2, padding=padding5)
        self.conv10 = nn.Conv1d(256, 256, 5, 2, padding=padding3)
        self.conv11 = nn.Conv1d(256, 256, 3, 2, padding=padding3)
        self.conv12 = nn.Conv1d(256, 256, 3, 2)
        self.fc1 = nn.Linear(768, 100)
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        R = nn.LeakyReLU(0.1)
        x = R(self.conv1(x))
        x = R(self.conv2(x))
        x = R(self.conv3(x))
        x = R(self.conv4(x))
        x = R(self.conv5(x))
        x = R(self.conv6(x))
        x = R(self.conv7(x))
        x = R(self.conv8(x))
        x = R(self.conv9(x))
        x = R(self.conv10(x))
        x = R(self.conv11(x))
        x = R(self.conv12(x))
        x = torch.flatten(x, start_dim=1)  # Start at dim1 such that the batch dimension is preserved.
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
