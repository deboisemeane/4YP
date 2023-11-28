from pathlib import Path
import pandas as pd
from .base_dataset import BaseDataset
import torch
import numpy as np


# Custom dataset for handcrafted frequency features.
class SHHSDataset_f(BaseDataset):

    def __init__(self, nsrrids: list[int], resample=None):
        data = []
        root_path = Path(__file__).parent.parent.parent
        for nsrrid in nsrrids:
            participant_data = pd.read_csv(root_path / f"data/Processed/shhs/Frequency_Features/nsrrid_{nsrrid}.csv")
            data.append(participant_data)
        self.data = pd.concat(data, axis=0)

        self.resample_factors = resample
        if resample is not None:
            self.resample()

        # Find label counts
        classes = [0, 1, 2, 3]
        label_counts = np.zeros(4)
        for i in classes:
            label_counts[i] = (self.data.iloc[:, -1] == float(i)).sum()
        self.label_counts = label_counts

        # Find weights which are inverse of label counts
        total_count = len(self.data)
        weight = total_count / label_counts
        weight = weight / weight.sum()  # Normalise weights
        weight = torch.tensor(weight).float()
        self.weight = weight
