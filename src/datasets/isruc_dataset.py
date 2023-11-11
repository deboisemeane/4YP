from pathlib import Path
import pandas as pd
import torch
import torch.utils.data as d


class ISRUCDataset(d.Dataset):    # This class is instantiated to select frequency feature data for chosen patients

    def __init__(self, patients, resample=None):

        data = []
        root_path = Path(__file__).parent.parent.parent
        for patient in patients:       # Patients are a list of patients which we want to include in this custom dataset.
            patient_data = pd.read_csv(root_path / f"data/Processed/ISRUC/Frequency_Features/patient_{patient}.csv")
            # Stick together dataframes of features for the selected patients
            data.append(patient_data)
        self.data = pd.concat(data, axis=0)

        self.resample_factors = resample
        if resample is not None:
            self.resample()

    # Appends resampled data to the dataset.
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
