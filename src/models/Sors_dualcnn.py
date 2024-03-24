import torch
import torch.nn as nn
import torch.nn.functional as F


class Sors_dualcnn(nn.Module):
    """
    Class intended to take x where the first 15000 items are the ecg samples, the next 3748 are upsampled rip for the same epoch.
    120s ECG at 125Hz = 15000
    120s RIP at 31.25Hz = 3748
    Total feature length = 18748
    """
    def __init__(self):
        super(Sors_dualcnn, self).__init__()

        # Convolutional layers for ECG
        self.ecg_conv1 = nn.Conv1d(1, 128, 7, 2, padding=3)
        self.ecg_conv2 = nn.Conv1d(128, 128, 7, 2, padding=3)
        self.ecg_conv3 = nn.Conv1d(128, 128, 7, 2, padding=3)
        self.ecg_conv4 = nn.Conv1d(128, 128, 7, 2, padding=2)
        self.ecg_conv5 = nn.Conv1d(128, 128, 7, 2, padding=2)
        self.ecg_conv6 = nn.Conv1d(128, 128, 7, 2, padding=3)
        self.ecg_conv7 = nn.Conv1d(128, 256, 7, 2, padding=3)
        self.ecg_conv8 = nn.Conv1d(256, 256, 5, 2, padding=2)
        self.ecg_conv9 = nn.Conv1d(256, 256, 5, 2, padding=2)
        self.ecg_conv10 = nn.Conv1d(256, 256, 5, 2, padding=1)
        self.ecg_conv11 = nn.Conv1d(256, 256, 3, 2, padding=1)
        self.ecg_conv12 = nn.Conv1d(256, 256, 3, 2, padding=1)
        # Batch Normalisation
        self.ecg_norm0 = nn.BatchNorm1d(1)
        self.ecg_norm1 = nn.BatchNorm1d(128)
        self.ecg_norm2 = nn.BatchNorm1d(128)
        self.ecg_norm3 = nn.BatchNorm1d(128)
        self.ecg_norm4 = nn.BatchNorm1d(128)
        self.ecg_norm5 = nn.BatchNorm1d(128)
        self.ecg_norm6 = nn.BatchNorm1d(128)
        self.ecg_norm7 = nn.BatchNorm1d(256)
        self.ecg_norm8 = nn.BatchNorm1d(256)
        self.ecg_norm9 = nn.BatchNorm1d(256)
        self.ecg_norm10 = nn.BatchNorm1d(256)
        self.ecg_norm11 = nn.BatchNorm1d(256)
        self.ecg_norm12 = nn.BatchNorm1d(256)

        # Convolutional layers for RIP
        self.rip_conv1 = nn.Conv1d(1, 128, 7, 2, padding=3)
        self.rip_conv2 = nn.Conv1d(128, 128, 7, 2, padding=3)
        self.rip_conv3 = nn.Conv1d(128, 128, 7, 2, padding=3)
        self.rip_conv4 = nn.Conv1d(128, 128, 7, 2, padding=2)
        self.rip_conv5 = nn.Conv1d(128, 128, 7, 2, padding=2)
        self.rip_conv6 = nn.Conv1d(128, 128, 7, 2, padding=3)
        self.rip_conv7 = nn.Conv1d(128, 256, 7, 2, padding=3)
        self.rip_conv8 = nn.Conv1d(256, 256, 5, 2, padding=2)
        self.rip_conv9 = nn.Conv1d(256, 256, 5, 2, padding=2)
        self.rip_conv10 = nn.Conv1d(256, 256, 5, 2, padding=1)
        self.rip_conv11 = nn.Conv1d(256, 256, 3, 1, padding=1)
        self.rip_conv12 = nn.Conv1d(256, 256, 3, 1)
        # Batch Normalisation
        self.rip_norm0 = nn.BatchNorm1d(1)
        self.rip_norm1 = nn.BatchNorm1d(128)
        self.rip_norm2 = nn.BatchNorm1d(128)
        self.rip_norm3 = nn.BatchNorm1d(128)
        self.rip_norm4 = nn.BatchNorm1d(128)
        self.rip_norm5 = nn.BatchNorm1d(128)
        self.rip_norm6 = nn.BatchNorm1d(128)
        self.rip_norm7 = nn.BatchNorm1d(256)
        self.rip_norm8 = nn.BatchNorm1d(256)
        self.rip_norm9 = nn.BatchNorm1d(256)
        self.rip_norm10 = nn.BatchNorm1d(256)
        self.rip_norm11 = nn.BatchNorm1d(256)
        self.rip_norm12 = nn.BatchNorm1d(256)

        # Define the MLP with the adjusted input size
        self.fc1 = nn.Linear(1280, 100)
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        R = nn.LeakyReLU(0.1)

        # Split the input
        ecg = x[:, :, :15000]  # ECG data with shape (batch_size, 1, 15000)
        rip = x[:, :, 15000:]  # RIP data with shape (batch_size, 1, 3748)

        # Process ECG data
        ecg = R(self.ecg_norm1(self.ecg_conv1(self.ecg_norm0(ecg))))
        ecg = R(self.ecg_norm2(self.ecg_conv2(ecg)))
        ecg = R(self.ecg_norm3(self.ecg_conv3(ecg)))
        ecg = R(self.ecg_norm4(self.ecg_conv4(ecg)))
        ecg = R(self.ecg_norm5(self.ecg_conv5(ecg)))
        ecg = R(self.ecg_norm6(self.ecg_conv6(ecg)))
        ecg = R(self.ecg_norm7(self.ecg_conv7(ecg)))
        ecg = R(self.ecg_norm8(self.ecg_conv8(ecg)))
        ecg = R(self.ecg_norm9(self.ecg_conv9(ecg)))
        ecg = R(self.ecg_norm10(self.ecg_conv10(ecg)))
        ecg = R(self.ecg_norm11(self.ecg_conv11(ecg)))
        ecg = R(self.ecg_norm12(self.ecg_conv12(ecg)))
        ecg = torch.flatten(ecg, start_dim=1)

        # Process RIP data
        rip = R(self.rip_norm1(self.rip_conv1(self.rip_norm0(rip))))
        rip = R(self.rip_norm2(self.rip_conv2(rip)))
        rip = R(self.rip_norm3(self.rip_conv3(rip)))
        rip = R(self.rip_norm4(self.rip_conv4(rip)))
        rip = R(self.rip_norm5(self.rip_conv5(rip)))
        rip = R(self.rip_norm6(self.rip_conv6(rip)))
        rip = R(self.rip_norm7(self.rip_conv7(rip)))
        rip = R(self.rip_norm8(self.rip_conv8(rip)))
        rip = R(self.rip_norm9(self.rip_conv9(rip)))
        rip = R(self.rip_norm10(self.rip_conv10(rip)))
        rip = R(self.rip_norm11(self.rip_conv11(rip)))
        rip = R(self.rip_norm12(self.rip_conv12(rip)))
        rip = torch.flatten(rip, start_dim=1)

        # Concatenate
        x = torch.cat((ecg, rip), dim=1)

        # MLP
        x = R(self.fc1(x))
        x = self.fc2(x)

        return x

# Example use
if __name__ == '__main__':
    import numpy as np
    x_test = np.zeros((64, 1, 18748))
    x_test = torch.tensor(x_test, dtype=torch.float32)
    model = Sors_dualcnn()
    print(model(x_test).shape)
