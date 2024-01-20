from pathlib import Path
import pandas as pd
from .base_dataset import BaseDataset
import torch
import numpy as np
from pathlib import Path
import csv


# Custom dataset for handcrafted frequency features.
class SHHSDataset_t:

    def __init__(self, nsrrids: list[int], data_dir: str):
        self.nsrrids = nsrrids
        self.data_dir = data_dir
        self.index_df = self.__create_index_df__()

        # Find label counts and then class weights (inversely proportional) to be provided to loss criterion.
        classes = [0, 1, 2, 3]
        label_counts = np.zeros(4)
        for i in classes:
            label_counts[i] = (self.index_df["labels"] == float(i)).sum()
        self.label_counts = label_counts

        total_count = len(self.index_df["labels"])
        weight = total_count / label_counts
        weight = weight / weight.sum()  # Normalise weights
        weight = torch.tensor(weight).float()
        self.weight = weight

    # Creates an index dataframe which specifies the nsrrid (and therefore filename),
    # and row number within that file of a training example.
    def __create_index_df__(self):
        nsrrids = []
        row_numbers = []
        labels = []

        data_dir = self.data_dir
        for nsrrid in self.nsrrids:
            data = np.load(file=data_dir/f"nsrrid_{nsrrid}.npy")
            for i in range(data.shape[0]):
                row_numbers.append(i)
                nsrrids.append(nsrrid)
                labels.append(int(data[i, -1]))

        index_dict = {"nsrrids": nsrrids, "row_numbers": row_numbers, "labels": labels}
        index_df = pd.DataFrame(index_dict)
        return index_df

    def __getitem__(self, idx):
        index_df = self.index_df
        nsrrid = index_df.loc[idx, "nsrrids"]
        row_number = index_df.loc[idx, "row_numbers"]
        label = index_df.loc[idx, "labels"]

        # Read the specific row
        data = np.load(file=self.data_dir / f"nsrrid_{nsrrid}.npy", mmap_mode='r')   # Open read-only to prevent so that only the specific row is loaded from disk
        features = data[row_number, 0:-1]   # Don't include the label

        # Convert to tensors
        label = torch.tensor(label, dtype=torch.long)  # CrossEntropyLoss expects integer type
        features = torch.tensor(features, dtype=torch.float32)  # Conv1d expects torch.float32 dtype
        features = features.unsqueeze(0)  # Conv1D expects shape [batchsize, n_channels, data_length].
                                          # If I don't specify n_channels=1, it will think the batch size is 1 and the n_channels is 64.
        return {"features": features, "label": label}

    def __len__(self):
        return self.index_df.shape[0]
