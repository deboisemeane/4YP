from pathlib import Path
import pandas as pd
import torch
import torch.utils.data as d
from .base_dataset_f import BaseDataset_f


class ISRUCDataset(BaseDataset_f):    # This class is instantiated to select frequency feature data for chosen patients

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


