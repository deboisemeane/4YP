import numpy as np
import pandas as pd
import torch
import torch.utils.data as d


class ISRUCDataset(d.Dataset):    # This class is instantiated to select frequency feature data for chosen patients

    def __init__(self, patients):
        self.data = []
        for patient in patients:       # Patients are a list of patients which we want to include in this custom dataset.
            patient_data = pd.read_csv(f"data/Processed/ISRUC/Frequency_Features/patient_{patient}.csv")
            # Stick together dataframes of features for the selected patients
            self.data.append(patient_data)
            self.data = pd.concat(self.data, axis=0)

    def __getitem__(self, idx):  # We have to overwrite the torch.utils.data.Datasets.__getitem__() method

        # Get column titles for the features (excludes labels)
        feature_columns = self.data.columns[1:-1]  # Excludes first column which is unnamed and contains indexes.
        features = self.data.iloc[idx][feature_columns]

        # Get column title for the labels
        label_column = self.data.columns[-1]
        label = self.data.iloc[idx][label_column]

        # Use one-hot encoding
        label_one_hot = np.zeros(4)
        label_one_hot[label] = 1    # N3:[1,0,0,0] N2/N1:[0,1,0,0] REM:[0,0,1,0] W:[0,0,0,1]

        # Convert to tensors
        label_one_hot = torch.tensor(label_one_hot, dtype=torch.long)
        features = torch.tensor(features, dtype=torch.float32)

        return {"features": features,
                "label": label_one_hot}

    def __len__(self):  # We have to overwrite the torch.utils.data.Datasets.__len__() method
            return len(self.data)
