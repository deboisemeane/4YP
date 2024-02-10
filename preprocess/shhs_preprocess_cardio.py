import mne
import os
import numpy as np
import pandas as pd
import scipy
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
        self.epochs_removed = []  # In the recordings that weren't rejected, how many 30s epochs have been excluded?

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

        five_minute_segments = []  # This will contain start times of segments > 5minutes long
        disconnects = []  # This will contain 1s (disconnected) or 0s (connected) indicating whether each 5minute segment is a disconnect
        five_minute_segment_indexes = []  # This will contain indexes of >5min segments in the list of ALL segment start times.

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
            else:
                five_minute_segments.append(segment_start)
                five_minute_segment_indexes.append(i)
            # Otherwise check if the segment is classified as a disconnect
            hr_segment = hr[segment_start:segment_end]
            hr_95_percentile = np.percentile(hr_segment, 95)
            hr_mean, hr_std = np.mean(hr_segment), np.std(hr_segment)
            if hr_95_percentile < 5 or hr_std > 30 or hr_mean < 25:
                disconnects.append(1)  # The >5minute segment is classified as a disconnect
            else:
                disconnects.append(0)  # The >5minute segment is not classified as a disconnect
        # Group disconnects and connects into chunks using diff
        disconnect_diff = list(np.diff(disconnects)) # Values of 1 indicate the disconnect chunk started at this >5min segment, -1 indicates a reconnect.
                                                # 0 means the equipment did not disconnect or reconnect at the start of this >5min segment.
        # First element isn't found by diff - insert the start of a disconnect chunk if the beginning is disconnected
        if disconnects[0] == 1:
            _ = 1
        else:
            _ = 0
        disconnect_diff = list(np.insert(disconnect_diff, 0, _))

        # If the equipment is disconnected at the end of the recording:
        if disconnects[-1] == 1:
            # Set t_end to be the time of the last disconnect
            i = len(disconnect_diff) - 1 - disconnect_diff[::-1].index(1)
            t_end = five_minute_segments[i]
            # Set boolean to tell us there is a disconnection at the end of the recording
            end_disconnected = True
        # Else t_end remains at the end of the recording and set boolean to say there is no disconnection at the end.
        else:
            end_disconnected = False
        # If the equipment is disconnected at the start of the recording:
        if disconnects[0] == 1:
            # Set t_start to be the time of the first reconnect
            t_start = five_minute_segments[disconnect_diff.index(-1)]
            # Set boolean to tell us there is a disconnection at the start of the recording.
            start_disconnected = True
        # Else t_start remains at the start of the recording and set boolean to say there is no disconnection at the start.
        else:
            start_disconnected = False
        # If the total number of disconnections is greater than the disconnections at the start or end
        if sum([_ == 1 for _ in disconnect_diff]) > (int(start_disconnected) + int(end_disconnected)):
            # We infer that there was a disconnection in the middle of the recording
            # Set t_start and t_end to None since we want to discard this recording
            t_start, t_end = None, None
            middle_disconnected = True
        else:
            middle_disconnected = False

        # Visual check that the algorithm does what we want it to.
        # Find all the disconnects and reconnects
        disconnect_indices = [index for (index, item) in enumerate(disconnect_diff) if item == 1]
        disconnect_times = [five_minute_segments[i] for i in disconnect_indices]
        reconnect_indices = [index for (index, item) in enumerate(disconnect_diff) if item == -1]
        reconnect_times = [five_minute_segments[i] for i in reconnect_indices]
        """if middle_disconnected: #or end_disconnected:
            fig, ax = plt.subplots()
            fig1, ax1 = plt.subplots()
            ax.plot(position.squeeze())
            ax1.plot(hr.squeeze())
            if t_start is not None:
                ax1.axvline(t_start, color='r', label='Start/End')
                ax1.axvline(t_end, color='r')
            for i,t in enumerate(disconnect_times):
                label = 'Disconnections' if i == 0 else None
                ax1.axvline(t, color='y', linewidth=1, label=label)
            for i,t in enumerate(reconnect_times):
                label = 'Reconnections' if i == 0 else None
                ax1.axvline(t, color='g', linewidth=1, label=label)
            ax1.legend()
            print("visualised")
            plt.close("all")"""
        if middle_disconnected:
            self.recordings_rejected += 1
        else:
            epochs_removed = np.ceil(t_start / 30) + np.floor(segments[-1]/30) - np.ceil(t_end / 30)
            self.epochs_removed.append(epochs_removed)
        return t_start, t_end

    def process(self, data_types: list, incl_preceeding_epochs=0, incl_following_epochs=0):
        """
        This function does preprocessing for cardiorespiratory data.
        :param data_types: A list which indicates the types of data we want: "THOR RES", "ECG", or both. If we select
        both, the raw objects will be created separately, so that THOR RES can be upsampled, and then recombined into a single raw object
        , and each row of the saved numpy array will consist of THOR RES concatenated with the ECG signal, and then
        a model can split the row and do with each as it sees fit.
        :param incl_preceeding_epochs: Context to include with each example.
        :param incl_following_epochs: Context to include with each example.
        :return: None: simply saves each nsrrid to a numpy file according to save_cardio_numpy
        """

        # Check valid number of preceeding and following epochs for each example.
        assert incl_preceeding_epochs >= 0, "Number of preceeding epochs to include with each example must be >= 0"
        assert incl_following_epochs >= 0, "Number of following epochs to include with each example must be >=0"

        for nsrrid in self.demographics["nsrrid"]:
            print(f"Processing nsrrid {nsrrid}")
            # Check for pulseox H.R. disconnections, which indicates equipment disconnection/ poor performance.
            t_start, t_end = self.remove_equipment_disconnect(nsrrid)
            print(f"Recordings rejected due to h.r. dropout in that wasn't at the start or end: {self.recordings_rejected}")
            # Skip this nsrrid if we decided to reject it based on equipment disconnects/ poor performance
            if t_start is None:
                continue
            # Load raw edf
            edfs_dir = 'data/Raw/shhs/polysomnography/edfs/shhs1/'
            patient_path = edfs_dir + "shhs1-" + str(nsrrid) + ".edf"
            raw_edfs = []
            if "ECG" in data_types:
                raw_ecg = mne.io.read_raw_edf(patient_path, include=["ECG"])
                raw = raw_ecg

            # Load raw respiratory data
            if "THOR RES" in data_types:
                raw_rip = mne.io.read_raw_edf(patient_path, include=["THOR RES"])
                raw_rip = self.upsample_rip(raw_rip)
                raw = raw_rip
                # Combining ecg and upsampled rip into one raw object
                if "ECG" in data_types:
                    raw = mne.io.concatenate_raws([raw_ecg, raw_rip])

            # Perform epoching, taking into account our new t_start and t_end.


    def upsample_rip(self, raw_rip: mne.io.Raw):
        """
        Creates a new mne.io.Raw object which is upsampled using cubic spline.
        :param raw_rip: the mne.io.Raw object we want to upsample.
        :return: raw_rip: the raw object reconstructed with upsampled data.
        """

        raw_data = raw_rip.get_data()[0]
        x = np.arange(len(raw_data))
        n_desired_samples = len(raw_data) * 125
        xc = 0 + np.arange(0, n_desired_samples) * (1/125)
        cs = scipy.interpolate.CubicSpline(x, raw_data)
        upsampled_data = cs(xc)
        info = mne.create_info(raw_rip.info['ch_names'], 125)
        raw_upsampled = mne.io.RawArray(upsampled_data, info)
        return raw_upsampled

    @staticmethod
    def load_stage_labels(self, nsrrid, raw_eeg) -> list:
        # Initialise labels array for one patient
        total_duration = raw_eeg.times[-1]
        n_epochs = total_duration // 30 + 1  # +1 because the last epoch is less than 30s, but we want to include it
        labels = [None] * int(n_epochs)
        # Create path string for current patient.
        annotations_path = f"data/Raw/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-{nsrrid}-nsrr.xml"
        # Check the annotations file is available for this patient.
        if os.path.isfile(annotations_path):
            annotations = ET.parse(annotations_path)
            root = annotations.getroot()
            # Check all ScoredEvents
            for event in root.findall(
                    ".//ScoredEvent"):  # . means starting from current node, // means ScoredEvent does not have to be a direct child of root
                event_type = event.find(".EventType").text
                # Check if this event is a stage annotation
                if "Stages|Stages" == event_type:
                    annotation = event.find("EventConcept").text
                    # Label integer is at end of EventConcept string.
                    label = int(annotation[-1])
                    # Convert to our own labelling convention ("0:N3, 1:N1/N2, 2:REM, 3:W")
                    if label == 3 or label == 4:  # Accounting for possibility of N4 in data.
                        label = 0
                    elif label == 2 or label == 1:
                        label = 1
                    elif label == 5:
                        label = 2
                    elif label == 0:
                        label = 3
                    else:
                        print(f"nsrrid: {nsrrid} Unsupported label '{annotation}'.")
                        label = None
                    # Store label in its corresponding positions in a numpy array, based on start and duration of this event.
                    start = float(event.find("Start").text)
                    duration = float(event.find("Duration").text)
                    for i in range(int(duration // 30)):
                        index = int(start // 30 + i)
                        # Sometimes the annotations go beyond the signal.
                        try:
                            labels[int(index)] = label
                        except IndexError:
                            pass
                            # This happens when we get to the last label, because the raw signal is not an exact length divisible by 30.
                            # There is one more label than there are full 30s epochs.
                            print(
                                f"nsrrid {nsrrid} Sleep stage annotation at epoch {index + 1} is out of range of EEG data ({n_epochs} epochs).")
        else:
            print(f"Annotations file for id {nsrrid} not available.")
        return labels

if __name__ == "__main__":
    os.chdir("C:/Users/Alex/PycharmProjects/4YP")
    pre = SHHSCardioPreprocessor()
    pre.process(["THOR RES", "ECG"])