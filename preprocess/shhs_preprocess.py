import mne
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import gc


# This class is used to process raw SHHS-1 data for all (selected) participants.
class SHHSPreprocessor:

    def __init__(self, **params):
        default_params = {"channel": "EEG",  # EEG gives C4-A1, EEG(sec) gives C3-A2
                          "lpf_cutoff": 30,
                          "feature_bin_freqwidth": 1,
                          "n_features": 20
                          }
        self.params = default_params
        self.params.update(params)

        self.demographics = pd.read_csv('data/Raw/shhs/datasets/shhs-harmonized-dataset-0.20.0.csv')
        self.choose_patients()  # Updates demographics DataFrame to only include acceptable examples.

        for nsrrid in self.demographics["nsrrid"]:
            raw_eeg = self.load_raw_eeg(self, nsrrid)
            stage = self.load_stage_labels(self, nsrrid, raw_eeg)
            lpf_epochs = self.create_lpf_epochs(raw_eeg, stage)
            freqs, fft_data = self.apply_hamming_fft(self, lpf_epochs)
            feature_freqs, features = self.select_freq_features(freqs, fft_data)
            self.save_features_labels_csv(nsrrid, features, lpf_epochs)

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

    # Creates the mne.io.Raw object for one nsrrid.
    @staticmethod
    def load_raw_eeg(self, nsrrid):
        edfs_dir = 'data/Raw/shhs/polysomnography/edfs/shhs1/'
        patient_path = edfs_dir+"shhs1-"+str(nsrrid)+".edf"
        channel = [self.params["channel"]]
        raw_eeg = mne.io.read_raw_edf(patient_path, include=channel)
        return raw_eeg

    # Returns a numpy array containing the sleep stage labels for nsrrid.
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
        else:
            print(f"Annotations file for id {nsrrid} not available.")
        return labels

    # Return a filtered mne.Epochs, with stage labels applied as metadata
    def create_lpf_epochs(self, raw_eeg: mne.io.Raw, stage: list) -> mne.Epochs:
        # d.c. detrend around 0
        detrend = 0
        # Stick on sleep stage as metadata.
        metadata = pd.DataFrame({"Sleep Stage": stage})
        # Create events and epochs
        events = mne.make_fixed_length_events(raw_eeg, duration=30)
        epochs = mne.Epochs(raw_eeg, events, tmin=0, tmax=30,
                            metadata=metadata, preload=True, detrend=detrend, baseline=None)
        # LPF
        lpf_epochs = epochs.copy().filter(l_freq=None, h_freq=self.params["lpf_cutoff"], method='fir',
                                          fir_design='firwin', phase='zero')
        return lpf_epochs

    # FFT
    @staticmethod
    def apply_hamming_fft(self, lpf_epochs: mne.Epochs) -> [np.ndarray, np.ndarray]:
        # Returns frequencies and corresponding absolute DFT components
        # Apply hamming window - reduces spectral leakage
        window_size = lpf_epochs.get_data().shape[2]  # Size = no. of points in each epoch
        hamming_window = np.hamming(window_size)

        # Apply FFT
        windowed_data = lpf_epochs.get_data() * hamming_window
        fft_data = np.abs(np.fft.rfft(windowed_data, axis=2))  # Shape (n_epochs, n_channels(1), epoch_length)

        # Create vector of corresponding frequencies
        sfreq = lpf_epochs.info["sfreq"]
        nyquist_freq = sfreq / 2  # Maximum frequency in the FFT
        endpoint = fft_data.shape[
                       2] % 2 == 0  # If number of points is even, the highest frequency is fs/2 (endpoint=True)
        freqs = np.linspace(0, nyquist_freq, len(fft_data[0][0]),
                            endpoint=endpoint)  # Evenly spaced from 0 to nyquist freq

        return freqs, fft_data

    # Average fft_data over bins
    def select_freq_features(self, freqs, fft_data):  # Averages fft data over specified frequency bins
        bin_freq_width = int(self.params["feature_bin_freqwidth"])  # Size of frequency range in each bin default=1Hz
        n_bins = int(self.params["n_features"])  # No. of bins

        n_epochs, n_channels = int(fft_data.shape[0]), int(fft_data.shape[1])
        binned_data = np.zeros((n_epochs, n_channels, n_bins))
        feature_freqs = np.zeros(n_bins)

        # Mean fft_data over frequency bins
        for i in range(0, n_bins):
            # Find frequency bound on the bin
            lfreq = i * bin_freq_width
            hfreq = (i + 1) * bin_freq_width
            # Get index of first frequency value greater than the bin lower bound.
            start_idx = next(x for x, val in enumerate(freqs) if val >= lfreq)
            # Get index of last frequency before exceeding the bin upper bound.
            end_idx = len(freqs) - next(x for x, val in enumerate(np.flip(freqs)) if val < hfreq) - 1
            # Average fft values over the current bin
            binned_data[:, :, i] = np.mean(fft_data[:, :, start_idx:end_idx], axis=2)
            # Find the centre frequency of the bin
            feature_freqs[i] = np.mean(freqs[start_idx:end_idx])

        # Normalise Features - energy of features in each epoch should sum to 1
        epoch_energies = np.sum(binned_data ** 2, axis=2)  # Summing over features - energy for each epoch
        epoch_energies = epoch_energies[:, :, np.newaxis]  # Add a new axis to make sure broadcasting works correctly.
        normalised_features = binned_data / np.sqrt(epoch_energies)
        print(np.sum(normalised_features ** 2, axis=2)) # Check that total energy of features in each epoch = 1
        return feature_freqs, normalised_features

    # Save features and labels to csv
    def save_features_labels_csv(self, nsrrid, features, epochs):  # Saves features and labels (where experts agree) to csv.
        # Find the examples that has a label (where experts agree) (Boolean Indexing)
        label_bool_idx = np.logical_not(epochs.metadata["Sleep Stage"].isna())
        # Select the features for the examples where the two experts agree, using boolean indexing.
        selected_features = features[label_bool_idx, :, :]
        # Select the labels for the same examples
        selected_labels = np.array(epochs.metadata["Sleep Stage"])[label_bool_idx]
        # Construct a dataframe containing the features and labels
        dataframe_columns = [str(feature_no) for feature_no in range(1, 1 + self.params["n_features"])] + ["Label"]
        data = np.concatenate((np.squeeze(selected_features), np.expand_dims(selected_labels, 1)), axis=1)
        df = pd.DataFrame(data, columns=dataframe_columns)
        # Write the dataframe to csv
        df.to_csv(f"data/Processed/shhs/Frequency_Features/nsrrid_{nsrrid}.csv")
        return