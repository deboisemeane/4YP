from pathlib import Path
import pandas as pd
from .base_dataset_f import BaseDataset_f


# Custom dataset for handcrafted frequency features.
class SHHSDataset_f(BaseDataset_f):

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

