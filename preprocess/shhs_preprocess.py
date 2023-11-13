import mne
import os
import numpy as np
import pandas as pd


class SHHSPreprocessor:

    def __init__(self, **params):

        self.demographics = pd.read_csv('data/Raw/shhs/datasets/shhs-harmonized-dataset-0.20.0.csv')
        self.choose_patients()
        #self.raw_eegs: [mne.io.Raw] = self.load_raw_eegs(self)

    # Exclude unwanted patients.
    def choose_patients(self):
        df = self.demographics
        full_scoring = df["nsrr_flag_spsw"] == 'full scoring'
        acceptable_ahi = df["nsrr_ahi_hp3r_aasm15"] <= 15
        first_visit = df["visitnumber"] == 1
        chosen_patients = np.logical_and(full_scoring, acceptable_ahi, first_visit)
        self.demographics = df[chosen_patients]




    @staticmethod
    def load_raw_eegs(self):
        raw_eegs = []
        edfs_dir = 'data/Raw/shhs/polysomnography/edfs/shhs1'
        file_list = os.listdir(edfs_dir)
        for file in file_list:
            raw_eegs.append(mne.io.read_raw_edf(edfs_dir + file))
        return raw_eegs

