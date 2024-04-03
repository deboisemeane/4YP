import torch
import torch.nn as nn
from src.models import Sors_hr


# This implements a modified version of the model used by Sors et al.
# It is designed to take feature length 3868 which corresponds to 4x 30s RIP @ 31.25Hz concat with 4x 30s H.R. @ 1Hz


class Sors_rip_hr(nn.Module):

    def __init__(self):
        super(Sors_rip_hr, self).__init__()

        # Convolution layers
        padding7 = 3
        padding5 = 2
        padding3 = 1

        # RIP Convolution
        self.rip_conv1 = nn.Conv1d(1, 128, 7, 2, padding=padding7)
        self.rip_conv2 = nn.Conv1d(128, 128, 7, 2, padding=padding7)
        self.rip_conv3 = nn.Conv1d(128, 128, 7, 2, padding=padding7)
        self.rip_conv4 = nn.Conv1d(128, 128, 7, 2, padding=padding5)
        self.rip_conv5 = nn.Conv1d(128, 128, 7, 2, padding=padding5)
        self.rip_conv6 = nn.Conv1d(128, 128, 7, 2, padding=padding7)
        self.rip_conv7 = nn.Conv1d(128, 256, 7, 2, padding=padding7)
        self.rip_conv8 = nn.Conv1d(256, 256, 5, 2, padding=padding5)
        self.rip_conv9 = nn.Conv1d(256, 256, 5, 2, padding=padding5)
        self.rip_conv10 = nn.Conv1d(256, 256, 5, 2, padding=padding3)
        self.rip_conv11 = nn.Conv1d(256, 256, 3, 1, padding=padding3)
        self.rip_conv12 = nn.Conv1d(256, 256, 3, 1, padding=padding3)

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

        # HR convolution
        self.hr_cnn = Sors_hr()

        # MLP
        self.fc1 = nn.Linear(1536, 100)
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        # Split the input
        rip = x[:, :, :3748]
        hr = x[:, :, 3748:]

        # RIP convolution
        R = nn.LeakyReLU(0.1)
        rip = R(self.norm1(self.rip_conv1(self.norm0(rip))))
        rip = R(self.norm2(self.rip_conv2(rip)))
        rip = R(self.norm3(self.rip_conv3(rip)))
        rip = R(self.norm4(self.rip_conv4(rip)))
        rip = R(self.norm5(self.rip_conv5(rip)))
        rip = R(self.norm6(self.rip_conv6(rip)))
        rip = R(self.norm7(self.rip_conv7(rip)))
        rip = R(self.norm8(self.rip_conv8(rip)))
        rip = R(self.norm9(self.rip_conv9(rip)))
        rip = R(self.norm10(self.rip_conv10(rip)))
        rip = R(self.norm11(self.rip_conv11(rip)))
        rip = R(self.norm12(self.rip_conv12(rip)))
        rip = torch.flatten(rip, start_dim=1)  # Start at dim1 such that the batch dimension is preserved.

        # HR convolution
        hr = self.hr_cnn(hr)

        x = torch.cat((rip, hr), dim=1)

        x = R(self.fc1(x))
        x = self.fc2(x)
        # Softmax is implemented in cross entropy loss, so we don't need it here.
        return x


if __name__ == '__main__':
    import numpy as np

    x_test = np.zeros((64, 1, 3868))
    x_test = torch.tensor(x_test, dtype=torch.float32)
    model = Sors_rip_hr()
    print(model(x_test).shape)
