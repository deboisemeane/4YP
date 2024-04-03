import torch
import torch.nn as nn


# This implements a modified version of the model used by Sors et al.
# It is designed to take feature length 120 which corresponds to 4x 30s HR @ 1Hz
# The CNN produces 768 features from this and does not include an MLP.

class Sors_hr(nn.Module):

    def __init__(self):
        super(Sors_hr, self).__init__()

        # Convolution layers
        padding7 = 3
        padding5 = 2
        padding3 = 1

        self.conv1 = nn.Conv1d(1, 128, 3, 2, padding=padding3)
        self.conv2 = nn.Conv1d(128, 128, 3, 2, padding=padding3)
        self.conv3 = nn.Conv1d(128, 128, 3, 2, padding=padding3)
        self.conv4 = nn.Conv1d(128, 256, 3, 2, padding=0)
        self.conv5 = nn.Conv1d(256, 256, 3, 2, padding=0)
        """
        self.conv6 = nn.Conv1d(256, 256, 3, 1, padding=padding3)
        self.conv7 = nn.Conv1d(256, 256, 3, 1, padding=padding3)
        self.conv8 = nn.Conv1d(256, 256, 3, 2, padding=padding3)
        self.conv9 = nn.Conv1d(256, 256, 3, 2, padding=padding3)
        self.conv10 = nn.Conv1d(256, 256, 3, 2, padding=padding3)
        self.conv11 = nn.Conv1d(256, 256, 3, 1, padding=padding3)
        self.conv12 = nn.Conv1d(256, 256, 3, 1, padding=padding3)
        """

        # Batch Normalisation
        self.norm0 = nn.BatchNorm1d(1)
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(128)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(256)
        self.norm5 = nn.BatchNorm1d(256)
        """
        self.norm6 = nn.BatchNorm1d(256)
        self.norm7 = nn.BatchNorm1d(256)
        self.norm8 = nn.BatchNorm1d(256)
        self.norm9 = nn.BatchNorm1d(256)
        self.norm10 = nn.BatchNorm1d(256)
        self.norm11 = nn.BatchNorm1d(256)
        self.norm12 = nn.BatchNorm1d(256)
        """

    def forward(self, x):
        R = nn.LeakyReLU(0.1)
        x = R(self.norm1(self.conv1(self.norm0(x))))
        x = R(self.norm2(self.conv2(x)))
        x = R(self.norm3(self.conv3(x)))
        x = R(self.norm4(self.conv4(x)))
        x = R(self.norm5(self.conv5(x)))
        """
        x = R(self.norm6(self.conv6(x)))
        x = R(self.norm7(self.conv7(x)))
        x = R(self.norm8(self.conv8(x)))
        x = R(self.norm9(self.conv9(x)))
        x = R(self.norm10(self.conv10(x)))
        x = R(self.norm11(self.conv11(x)))
        x = R(self.norm12(self.conv12(x)))
        """
        x = torch.flatten(x, start_dim=1)  # Start at dim1 such that the batch dimension is preserved.

        # Softmax is implemented in cross entropy loss, so we don't need it here.
        return x


if __name__ == '__main__':
    import numpy as np
    x_test = np.zeros((64, 1, 120))
    x_test = torch.tensor(x_test, dtype=torch.float32)
    model = Sors_hr()
    print(model(x_test).shape)
