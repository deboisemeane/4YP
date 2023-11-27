import pandas as pd
import torch
import torch.utils.data as d
import numpy as np


# Base class for frequency feature datasets.

class BaseDataset_f(d.Dataset):
    resample_factors: dict[str:float]
    data: pd.DataFrame
    label_counts: np.ndarray  # Number of examples for each label 0, 1, 2, 3
    weight: torch.tensor  # Weight for loss criterion, based on inverse of label counts

    # Resample factors should be specified in a dictionary {"label": factor}
    def resample(self):
        df = self.data
        for stage, factor in self.resample_factors.items():
            if factor > 1:
                stage_data = df[df.iloc[:, 21] == float(stage)]
                num_samples_to_add = int(len(stage_data) * (factor - 1))
                # Take some random samples of those corresponding to the current stage.
                resamples = stage_data.sample(num_samples_to_add, replace=True)
                df = pd.concat([df, resamples], axis=0)
            else:
                raise ValueError("Resample factor should be > 1.")
        self.data = df

    def __getitem__(self, idx):  # We have to overwrite the torch.utils.data.Datasets.__getitem__() method

        # Get column titles for the features (excludes labels)
        feature_columns = self.data.columns[1:-1]  # Excludes first column which is unnamed and contains indeces.
        features = self.data.iloc[idx, self.data.columns.get_loc(feature_columns[0]):self.data.columns.get_loc(feature_columns[-1]) + 1]

        # Get column title for the labels
        label_column = self.data.columns[-1]
        label = self.data.iloc[idx, self.data.columns.get_loc(label_column)]

        # Convert to tensors
        label = torch.tensor(label, dtype=torch.long)   # CrossEntropyLoss expects integer type
        features = torch.tensor(features.values, dtype=torch.float32)

        return {"features": features,
                "label": label}

    def __len__(self):  # We have to overwrite the torch.utils.data.Datasets.__len__() method
            return len(self.data)