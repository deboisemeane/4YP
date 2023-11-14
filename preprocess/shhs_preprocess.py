import mne
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import gc


# This class is used to process raw SHHS-1 data for all (selected) participants.
class SHHSPreprocessor:

    def __init__(self, **params):
        default_params = {"channel": "EEG",  # EEG: C4-A1, EEG(sec): C3-A2
                          "lpf_cutoff": 30,
                          "feature_bin_freqwidth": 1,
                          "n_features": 20
                          }
        self.params = default_params
        self.params.update(params)

        self.demographics = pd.read_csv('data/Raw/shhs/datasets/shhs-harmonized-dataset-0.20.0.csv')
        self.choose_patients()  # Updates demographics DataFrame to only include acceptable examples.
        self.raw_eegs: {mne.io.Raw} = self.load_raw_eegs(self)
        self.stage: {np.ndarray} = self.load_stage_labels(self)
        self.create_and_save_lpf_epochs()

    # Exclude unwanted patients.
    # Sleep disorders, second visits, and unavailable sleep scoring are excluded.
    def choose_patients(self):
        df = self.demographics
        full_scoring = df["nsrr_flag_spsw"] == 'full scoring'
        acceptable_ahi = df["nsrr_ahi_hp3r_aasm15"] <= 15
        first_visit = df["visitnumber"] == 1
        chosen_patients = np.logical_and(full_scoring, acceptable_ahi, first_visit)
        df = df[chosen_patients]
        # Check eeg data is available for chosen patients.
        edfs_dir = 'data/Raw/shhs/polysomnography/edfs/shhs1/'
        unavailable_eeg_indeces = []
        for i, nsrrid in enumerate(df["nsrrid"]):
            edf_path = edfs_dir + "shhs1-" + str(nsrrid) + ".edf"
            if np.logical_not(os.path.isfile(edf_path)):
                unavailable_eeg_indeces.append(i)
                print(f"EEG not available for nsrrid {nsrrid}.")
        # Since we dropped previous unacceptable participants, the indices no longer match the rows, so we need to reset.
        df = df.reset_index(drop=True)
        df = df.drop(unavailable_eeg_indeces, axis='index')
        self.demographics = df

    # Returns a dict {"nsrrid": mne.io.Raw}
    @staticmethod
    def load_raw_eegs(self):
        raw_eegs = {}
        edfs_dir = 'data/Raw/shhs/polysomnography/edfs/shhs1/'
        chosen_patients = self.demographics["nsrrid"]

        for patient in chosen_patients:
            patient_path = edfs_dir+"shhs1-"+str(patient)+".edf"
            channel = self.params["channel"]
            raw_eeg = mne.io.read_raw_edf(patient_path, include=channel)
            raw_eegs.update({str(patient): raw_eeg})
        return raw_eegs

    # Returns a dictionary of numpy arrays for 30s epoch labels. {"nsrrid": labels}
    @staticmethod
    def load_stage_labels(self):
        stage = {}
        # Iterate over all the selected patients.
        for nsrrid in self.demographics["nsrrid"]:
            # Initialise labels array for one patient
            total_duration = self.raw_eegs[f'{nsrrid}'].times[-1]
            n_epochs = total_duration // 30 + 1  # +1 because the last epoch is less than 30s, but we want to include it
            labels = [None] * int(n_epochs)
            # Create path string for current patient.
            annotations_path = f"data/Raw/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-{nsrrid}-nsrr.xml"
            # Check the annotations file is available for this patient.
            if os.path.isfile(annotations_path):
                annotations = ET.parse(annotations_path)
                root = annotations.getroot()
                # Check all ScoredEvents
                for event in root.findall(".//ScoredEvent"):    # . means starting from current node, // means ScoredEvent does not have to be a direct child of root
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
                            # Sometimes the annotations go beyond the EEG signal.
                            try:
                                labels[int(index)] = label
                            except IndexError:
                                pass
                                # This happens when we get to the last label, because the raw EEG is not an exact length divisible by 30.
                                # There is one more label than there are full 30s EEG epochs.
                                print(f"nsrrid {nsrrid} Sleep stage annotation at epoch {index+1} is out of range of EEG data ({n_epochs} epochs).")
                stage.update({f"{nsrrid}": labels})
            else:
                print(f"Annotations file for id {nsrrid} not available.")
        return stage

        # Return a dict of mne.Epochs objects. {"nsrrid": mne.Epochs}
    def create_and_save_lpf_epochs(self):
        d = {}
        detrend = 0  # d.c. detrend around 0
        save_dir = "data/Processed/shhs/Filtered Epochs/"  # This is where we will save filtered epochs objects
        # Create an epochs object for each participant we selected earlier.
        for nsrrid in self.demographics["nsrrid"]:
            # Get sleep stage labels we will stick on as metadata.
            sleep_stage = self.stage[f"{nsrrid}"]
            metadata = pd.DataFrame({"Sleep stage": sleep_stage})
            # Get raw eeg
            raw_eeg = self.raw_eegs[f"{nsrrid}"]
            # Create events and epochs
            events = mne.make_fixed_length_events(raw_eeg, duration=30)
            epochs = mne.Epochs(raw_eeg, events, tmin=0, tmax=30,
                                metadata=metadata, preload=True, detrend=detrend, baseline=None)
            # LPF
            lpf_epochs = epochs.copy().filter(l_freq=None, h_freq=self.params["lpf_cutoff"], method='fir',
                                              fir_design='firwin', phase='zero')
            # Save the filtered epochs to disk
            file_name = os.path.join(save_dir, f"filtered_epochs_nsrrid_{nsrrid}.fif")
            lpf_epochs.save(file_name, overwrite=True)

            del epochs, lpf_epochs
            gc.collect()

    @staticmethod
    def load_filtered_epochs(self, nsrrid):
        file_name = f"path/to/your/saved_epochs_directory/filtered_epochs_nsrrid_{nsrrid}.fif"
        if os.path.exists(file_name):
            return mne.read_epochs(file_name, preload=True)
        else:
            print(f"File not found: {file_name}")
            return None



