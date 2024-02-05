import torch
import torch.nn as nn


# This implements a modified version of the model used by Sors et al.
# It is designed to take feature length 3750 which corresponds to one 30s epoch at 125Hz, i.e. without surrounding context.
class Sors_nocontext2(nn.Module):

    def __init__(self):
        super(Sors_nocontext2, self).__init__()

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
        self.conv8 = nn.Conv1d(256, 256, 5, 2, padding=padding5)
        self.conv9 = nn.Conv1d(256, 256, 5, 2, padding=padding5)
        self.conv10 = nn.Conv1d(256, 256, 5, 2, padding=padding3)
        self.conv11 = nn.Conv1d(256, 256, 3, 1, padding=padding3)
        self.conv12 = nn.Conv1d(256, 256, 3, 1, padding=padding3)
        self.fc1 = nn.Linear(768, 100)
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
        self.norm8 = nn.BatchNorm1d(256)
        self.norm9 = nn.BatchNorm1d(256)
        self.norm10 = nn.BatchNorm1d(256)
        self.norm11 = nn.BatchNorm1d(256)
        self.norm12 = nn.BatchNorm1d(256)

    def forward(self, x):
        R = nn.LeakyReLU(0.1)
        x = R(self.norm1(self.conv1(self.norm0(x))))
        x = R(self.norm2(self.conv2(x)))
        x = R(self.norm3(self.conv3(x)))
        x = R(self.norm4(self.conv4(x)))
        x = R(self.norm5(self.conv5(x)))
        x = R(self.norm6(self.conv6(x)))
        x = R(self.norm7(self.conv7(x)))
        x = R(self.norm8(self.conv8(x)))
        x = R(self.norm9(self.conv9(x)))
        x = R(self.norm10(self.conv10(x)))
        x = R(self.norm11(self.conv11(x)))
        x = R(self.norm12(self.conv12(x)))
        x = torch.flatten(x, start_dim=1)  # Start at dim1 such that the batch dimension is preserved.
        x = R(self.fc1(x))
        x = self.fc2(x)
        # Softmax is implemented in cross entropy loss, so we don't need it here.
        return x


if __name__ == '__main__':
    import numpy as np
    x_test = np.zeros((64, 1, 3750))
    x_test = torch.tensor(x_test, dtype=torch.float32)
    model = Sors_nocontext2()
    print(model(x_test).shape)
