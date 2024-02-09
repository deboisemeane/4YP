import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
import yasa
from utils import get_data_dir_shhs


# This class is used to process raw SHHS-1 data for all (selected) participants.
class SHHSCardioPreprocessor:

    def __init__(self, **params):
        default_params = {"lpf": True,       # Decide whether to low pass filter
                          "lpf_cutoff": 30,
                          }
        self.params = default_params
        self.params.update(params)

        self.demographics = pd.read_csv('data/Raw/shhs/datasets/shhs-harmonized-dataset-0.20.0.csv')
        self.choose_patients()  # Updates demographics DataFrame to only include acceptable examples.
        self.recordings_rejected = 0  # Recordings rejected due to equipment removal >5minutes long in the middle of the night.
        self.epochs_removed = 0  # In the recordings that weren't rejected, how many 30s epochs have been excluded?
        self.total_epochs_processed = 0  # We will use this to work out the proportion of epochs excluded due to equiment disconnect.

    def choose_patients(self):
        df = self.demographics
        full_scoring = df["nsrr_flag_spsw"] == 'full scoring'
        acceptable_ahi = df["nsrr_ahi_hp3r_aasm15"] <= 15
        first_visit = df["visitnumber"] == 1
        chosen_patients = np.logical_and(full_scoring, acceptable_ahi, first_visit)
        df = df[chosen_patients]

        # Check edf file is available for chosen patients.
        edfs_dir = 'data/Raw/shhs/polysomnography/edfs/shhs1/'
        unavailable_edf_indeces = []
        for i, nsrrid in enumerate(df["nsrrid"]):
            edf_path = edfs_dir + "shhs1-" + str(nsrrid) + ".edf"
            if np.logical_not(os.path.isfile(edf_path)):
                unavailable_edf_indeces.append(i)
                print(f"EDF not available for nsrrid {nsrrid}.")
        # Since we dropped previous unacceptable participants, the indices no longer match the rows, so we need to reset.
        df = df.reset_index(drop=True)
        df = df.drop(unavailable_edf_indeces, axis='index')
        self.demographics = df

    @staticmethod
    def load_mne_raw(self, nsrrid):
        edfs_dir = 'data/Raw/shhs/polysomnography/edfs/shhs1/'
        patient_path = edfs_dir+"shhs1-"+str(nsrrid)+".edf"
        channel = ["THOR RES", "ECG", "H.R.", "POSITION"]
        raw = mne.io.read_raw_edf(patient_path, include=channel)
        return raw

    # Function to give a new start and end time based on equipment disconnections.
    # If there is equipment disconnect >5 minutes long, not at the start or end, then the whole recording is thown out.
    def remove_equipment_disconnect(self, nsrrid):
        edfs_dir = 'data/Raw/shhs/polysomnography/edfs/shhs1/'
        patient_path = edfs_dir + "shhs1-" + str(nsrrid) + ".edf"
        # Load channels separately due to differing sampling rates, otherwise MNE will resample lower sampling rates.
        (position, times) = mne.io.read_raw_edf(patient_path, include=["POSITION"]).get_data(return_times=True)
        hr = mne.io.read_raw_edf(patient_path, include=["H.R."]).get_data(return_times=True)[0]
        position, hr = position.squeeze(), hr.squeeze()
        # Segment the night based on changing position
        segments = np.nonzero(np.diff(position))[0]  # These are starting indeces corresponding to each position segment.
        segments = np.insert(segments, 0, 0)  # Need to include the start of the first segment, which isn't picked up by diff.
        # Decide whether each segment has equipment disconnect
        disconnects = []
        # Default start and end of recording if there are no disconnects
        t_start, t_end = times[0], times[-1]+1
        for i, segment_start in enumerate(segments):
            # Don't bother checking equipment disconnects if the segment is < 5minutes long.
            if i < len(segments)-1:
                segment_end = segments[i+1]
            else:
                segment_end = int(t_end)
            segment_length = segment_end - segment_start
            if segment_length < 5*60:
                continue
            # Otherwise check if h.r. in this segment is non-zero
            hr_segment = hr[segment_start:segment_end]
            hr_95_percentile = np.percentile(hr_segment, 95)
            hr_mean, hr_std = np.mean(hr_segment), np.std(hr_segment)
            if hr_95_percentile < 5 or hr_std < 0.1:
                disconnects.append(i)
                if segment_start == 0:
                    t_start = segments[i+1]
                if segment_start == segments[-1]:
                    t_end = segments[i]
                else:
                    t_start, t_end = None, None
                    break_start, break_end = segment_start, segment_end
                    self.recordings_rejected += 1
                    break

        # Visual check that the algorithm does what we want it to.
        if t_start!=0 or t_end!=times[-1]+1 or t_start is None:
            fig, ax = plt.subplots()
            fig1, ax1 = plt.subplots()
            ax.plot(position.squeeze())
            ax1.plot(hr.squeeze())
            if t_start is not None:
                ax1.axvline(t_start, color='r')
                ax1.axvline(t_end, color='r')
            else:
                ax1.axvline(break_start, color='r')
                ax1.axvline(break_end, color='r')
            print("visualised")
            plt.close("all")
        return t_start, t_end
    def process(self, data_types: list, incl_preceeding_epochs=0, incl_following_epochs=0):

        # Check valid number of preceeding and following epochs for each example.
        assert incl_preceeding_epochs >= 0, "Number of preceeding epochs to include with each example must be >= 0"
        assert incl_following_epochs >= 0, "Number of following epochs to include with each example must be >=0"

        for nsrrid in self.demographics["nsrrid"]:
            print(f"Processing nsrrid {nsrrid}")
            # Check for pulseox H.R. disconnections, which indicates equipment removal.
            t_start, t_end = self.remove_equipment_disconnect(nsrrid)
            print(f"Recordings rejected due to h.r. dropout in that wasn't at the start or end: {self.recordings_rejected}")


if __name__ == "__main__":
    os.chdir("C:/Users/Alex/PycharmProjects/4YP")
    pre = SHHSCardioPreprocessor()
    pre.process(["THOR RES", "ECG"])