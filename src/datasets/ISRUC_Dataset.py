
import pandas as pd
import torch
import torch.utils.data as d

class ISRUC_Dataset(d.Dataset):    # This class is instantiated to select frequency feature data for chosen patients

    def __init__(self, patients):
        self.data = []
        for patient in patients:       # Patients are a list of patients which we want to include in this custom dataset.
            patient_data = pd.read_csv(f"data/Processed/ISRUC/Frequency_Features/patient_{patient}.csv")
            self.data.append(patient_data)
            self.data = pd.concat(self.data, axis=0)

    def __getitem__(self, idx):  # We have to overwrite the torch.utils.data.Datasets.__getitem__() method
        features = self.data.iloc[idx].[0:20] ########### HOW TO INDEX COLUMNS BY INTEGER INSTEAD OF LABEL
        ##### OTHERWISE I NEED TO CHANGE COLUMN NAMES OF PROCESSED FEATURES TO JUST INTEGERS SO THAT I CAN SLICE "1":"20"
        # or something like that?
        return

    def __len__(self):  # We have to overwrite the torch.utils.data.Datasets.__len__() method