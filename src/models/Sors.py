import torch
import torch.nn as nn


# This implements the model used by Sors et al.
# It is designed to take feature length 1500, otherwise the fully connected layers will not work.
class Sors(nn.Module):

    def __init__(self):
        super(Sors, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 7, 2)
        self.conv2 = nn.Conv1d(128, 128, 7, 2)
        self.conv3 = nn.Conv1d(128, 128, 7, 2)
        self.conv4 = nn.Conv1d(128, 128, 7, 2)
        self.conv5 = nn.Conv1d(128, 128, 7, 2)
        self.conv6 = nn.Conv1d(128, 128, 7, 2)
        self.conv7 = nn.Conv1d(128, 256, 7, 2)
        self.conv8 = nn.Conv1d(256, 256, 5, 2)
        self.conv9 = nn.Conv1d(256, 256, 5, 2)
        self.conv10 = nn.Conv1d(256, 256, 5, 2)
        self.conv11 = nn.Conv1d(256, 256, 3, 2)
        self.conv12 = nn.Conv1d(256, 256, 3, 2)
        self.fc1 = nn.Linear(768, 100)
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        F = nn.LeakyReLU(0.1)
        x = F(self.conv1(x))
        x = F(self.conv2(x))
        x = F(self.conv3(x))
        x = F(self.conv4(x))
        x = F(self.conv5(x))
        x = F(self.conv6(x))
        x = F(self.conv7(x))
        x = F(self.conv8(x))
        x = F(self.conv9(x))
        x = F(self.conv10(x))
        x = F(self.conv11(x))
        x = F(self.conv12(x))
        x = torch.flatten(x, start_dim=1)  # Start at dim1 such that the batch dimension is preserved.
        x = F(self.fc1(x))
        x = self.fc2(x)
        # Softmax is implemented in cross entropy loss, so we don't need it here.
        return x
