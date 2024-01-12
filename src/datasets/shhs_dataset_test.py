from pathlib import Path
import pandas as pd
from .base_dataset import BaseDataset
import torch
import numpy as np
import csv


# Custom dataset for handcrafted frequency features.
class SHHSDataset_new:

    def __init__(self, nsrrids: list[int], data_dir: str, resample=None):
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
            df = pd.read_csv(data_dir / f"nsrrid_{nsrrid}.csv")
            label_column = df.columns[-1]
            for i in range(df.shape[0]):
                row_numbers.append(i)
                nsrrids.append(nsrrid)
                labels.append(df.iloc[i, df.columns.get_loc(label_column)])

        index_dict = {"nsrrids": nsrrids, "row_numbers": row_numbers, "labels": labels}
        index_df = pd.DataFrame(index_dict)
        return index_df

    def __getitem__(self, idx):
        index_df = self.index_df
        nsrrid = index_df.loc[idx, "nsrrids"]
        row_number = index_df.loc[idx, "row_numbers"]
        label = index_df.loc[idx, "labels"]

        # Calculate the number of rows to skip
        skip = row_number if row_number > 0 else None

        # Read the specific row
        df = pd.read_csv(self.data_dir / f"nsrrid_{nsrrid}.csv", skiprows=skip, nrows=1)
        features = df.iloc[0, 1:-1]  # Since we read only one row, we use iloc[0]

        # Convert to tensors
        label = torch.tensor(label, dtype=torch.long)  # CrossEntropyLoss expects integer type
        features = torch.tensor(features.values, dtype=torch.float32)
        return {"features": features, "label": label}

    def __len__(self):
        return self.index_df.shape[0]
