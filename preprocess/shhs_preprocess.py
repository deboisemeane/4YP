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
class SHHSPreprocessor:

    def __init__(self, **params):
        default_params = {"channel": "EEG",  # EEG gives C4-A1, EEG(sec) gives C3-A2
                          "lpf": True,       # Decide whether to low pass filter
                          "lpf_cutoff": 30,
                          "feature_bin_freqwidth": 1,
                          "n_features": 20,
                          "art_rejection": True
                          }
        self.params = default_params
        self.params.update(params)

        # Check if we're on the IBME cluster
        if Path('/data/wadh6184/').is_dir():
            self.root_dir = Path("/data/wadh6184/")
        # Otherwise use local directory for processed data
        else:
            self.root_dir = Path(__file__).parent.parent
        self.demographics = pd.read_csv(self.root_dir / 'data/Raw/shhs/datasets/shhs-harmonized-dataset-0.20.0.csv')
        self.choose_patients()  # Updates demographics DataFrame to only include acceptable examples.

    # Generates and saves to npy time domain inputs and labels.
    def process_t(self, incl_preceeding_epochs: int = 0, incl_following_epochs: int = 0):

        # Check valid number of preceeding and following epochs for each example.
        assert incl_preceeding_epochs >= 0, "Number of preceeding epochs to include with each example must be >= 0"
        assert incl_following_epochs >= 0, "Number of following epochs to include with each example must be >=0"

        art_rejection = self.params["art_rejection"]
        rejections = 0  # We will count the number of recordings rejected due to artefacts.
        for nsrrid in self.demographics["nsrrid"]:
            print(f"Processing nsrrid: {nsrrid}")
            raw_eeg = self.load_raw_eeg(self, nsrrid)
            stage = self.load_stage_labels(self, nsrrid, raw_eeg)

            # Artefact rejection
            if art_rejection is True:
                reject = self.std_rejection(raw_eeg, stage)
                if reject is True:
                    rejections += 1
                    continue  # Skips to next nsrrid

            # Low pass filter and create mne.Epochs object with stage label metadata.
            epochs = self.create_epochs(raw_eeg, stage)
            # Save the time features to npy
            self.save_t_features_labels_npy(nsrrid, epochs, incl_preceeding_epochs=incl_preceeding_epochs,
                                            incl_following_epochs=incl_following_epochs)
            if art_rejection is True:
                print(f"{rejections} recordings rejected due to >2% of epochs being artefacts.")

    # Generates and saves to csv frequency domain features and labels.
    def process_f(self):
        art_rejection = self.params["art_rejection"]
        rejections = 0
        for nsrrid in self.demographics["nsrrid"]:
            raw_eeg = self.load_raw_eeg(self, nsrrid)
            stage = self.load_stage_labels(self, nsrrid, raw_eeg)

            # Artefact rejection
            if art_rejection is True:
                reject = self.std_rejection(raw_eeg, stage)
                if reject is True:
                    rejections += 1
                    continue  # Skips to next nsrrid

            lpf_epochs = self.create_epochs(raw_eeg, stage)
            freqs, fft_data = self.apply_hamming_fft(self, lpf_epochs)
            #avg_stage_power_fractions, power_fractions = self.get_power_fractions(freqs, fft_data, lpf_epochs)
            #self.plot_power_fractions(avg_stage_power_fractions)

            feature_freqs, features = self.select_freq_features(freqs, fft_data)
            self.save_f_features_labels(nsrrid, features, lpf_epochs)

        if art_rejection is True:
            print(f"{rejections} recordings rejected due to >2% of epochs being artefacts.")

    # Sleep disorders, second visits, and unavailable sleep scoring are excluded.
    def choose_patients(self):

        df = self.demographics
        full_scoring = df["nsrr_flag_spsw"] == 'full scoring'
        acceptable_ahi = df["nsrr_ahi_hp3r_aasm15"] <= 15
        first_visit = df["visitnumber"] == 1
        chosen_patients = np.logical_and(full_scoring, acceptable_ahi, first_visit)
        df = df[chosen_patients]
        # Check eeg data is available for chosen patients.
        edfs_dir = self.root_dir / 'data/Raw/shhs/polysomnography/edfs/shhs1/'
        unavailable_eeg_indeces = []
        for i, nsrrid in enumerate(df["nsrrid"]):
            edf_path = edfs_dir / f"shhs1-{nsrrid}.edf"
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
        edfs_dir = self.root_dir / 'data/Raw/shhs/polysomnography/edfs/shhs1/'
        patient_path = edfs_dir / f"shhs1-{nsrrid}.edf"
        channel = [self.params["channel"]]
        raw_eeg = mne.io.read_raw_edf(patient_path, include=channel)
        return raw_eeg

    # Returns a list containing the sleep stage labels for nsrrid.
    @staticmethod
    def load_stage_labels(self, nsrrid, raw_eeg) -> list:
        # Initialise labels array for one patient
        total_duration = raw_eeg.times[-1]
        n_epochs = total_duration // 30 + 1  # +1 because the last epoch is less than 30s, but we want to include it
        labels = [None] * int(n_epochs)
        # Create path string for current patient.
        annotations_path = self.root_dir / f"data/Raw/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-{nsrrid}-nsrr.xml"
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

    # Standard deviation based artifact rejection using YASA. Returns a bool deciding to reject the recording.
    @staticmethod
    def std_rejection(data: mne.io.Raw, stage: list) -> bool:
        data_np = data.get_data()
        # Make sure data and hypnogram have same number of samples - this is required by yasa.art_detect
        n_samples = data.get_data().shape[1]
            # Repeat each stage label to match number of eeg samples
        sf = data.info["sfreq"]
        samples_per_window = 30 * sf
        hypno = [x for x in stage for _ in range(int(samples_per_window))]
            # Remove excess points for final window which is <30s
        excess = int(n_samples % samples_per_window)
        if excess != 0:
            hypno = hypno[0:-excess]

        hypno_np = np.array(hypno)

        # Remove samples where no label is given, to prevent yasa.art_detect from breaking
        # Create a mask where None values are marked as False
        mask = hypno_np != None

        # Filter both arrays using the mask
        hypno = hypno_np[mask]
        data = np.squeeze(data_np)[mask]

        art_epochs, zscores = yasa.art_detect(data=data, window=30, hypno=hypno, sf=sf, include=(0, 1, 2), method='std')
        # Reject recording if >2% of epochs are artifacts
        if sum(art_epochs) / len(art_epochs) > 0.02:
            reject = True
        else:
            reject = False
        return reject

    # Return a mne.Epochs object, with stage labels applied as metadata
    def create_epochs(self, raw_eeg: mne.io.Raw, stage: list) -> mne.Epochs:
        # Find sample frequency and length of one sample
        sf = raw_eeg.info["sfreq"]
        t_s = 1/sf

        # Apply d.c. detrend around 0
        detrend = 0
        # Stick on sleep stage as metadata.
        metadata = pd.DataFrame({"Sleep Stage": stage})
        # Create events and epochs
        events = mne.make_fixed_length_events(raw_eeg, duration=30)
        epochs = mne.Epochs(raw_eeg, events, tmin=0, tmax=30-t_s,
                            metadata=metadata, preload=True, detrend=detrend, baseline=None)
        # LPF
        if self.params["lpf"] is True:
            epochs = epochs.copy().filter(l_freq=None, h_freq=self.params["lpf_cutoff"], method='fir',
                                          fir_design='firwin', phase='zero', verbose='WARNING')
        return epochs

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
        # print(np.sum(normalised_features ** 2, axis=2)) # Check that total energy of features in each epoch = 1
        return feature_freqs, normalised_features

    def get_power_fractions(self, freqs, fft_data, lpf_epochs):
        labels = lpf_epochs.metadata["Sleep Stage"]
        # Epoch indices for corresponding stages N3 N2/N1 REM W
        stage_idx = np.zeros((4, len(labels)))
        for i in range(4):
            stage_idx[i] = labels == i
        # Frequency range indices for corresponding clinical bands
        delta_idx = np.logical_and(np.greater_equal(freqs, 0.5), np.less(freqs, 4))
        theta_idx = np.logical_and(np.greater_equal(freqs, 4), np.less(freqs, 8))
        alpha_idx = np.logical_and(np.greater_equal(freqs, 8), np.less(freqs, 12))
        beta_idx = np.logical_and(np.greater_equal(freqs, 12), np.less(freqs, 30))
        # Power fractions for each window
        # Delta, theta, alpha, beta
        delta_fft = fft_data * delta_idx  # Shape (n_windows, n_channels, n_freqs), zeros outside the desired frequency range
        theta_fft = fft_data * theta_idx
        alpha_fft = fft_data * alpha_idx
        beta_fft = fft_data * beta_idx

        fft = np.squeeze(np.array([delta_fft, theta_fft, alpha_fft, beta_fft]))  # Shape (n_bands, n_epochs, n_freq)

        powers = np.sum(fft * fft, axis=2)  # Total power for each window, shape (n_bands, n_epochs)

        power_fractions = powers / np.sum(powers, axis=0)  # Power fractions for each window, shape (n_bands, n_epochs)

        # Powers for each sleep stage, for each window
        stage_powers = np.flip(np.expand_dims(powers, 1) * np.expand_dims(stage_idx, 0),
                               axis=1)  # Shape (n_bands (4), n_stages (4), n_windows)
        # axis0: bands delta->beta
        # axis1: stages W->N3

        # Average powers for each sleep stage
        avg_stage_powers = np.average(stage_powers,
                                      axis=2)  # Averaged over windows corresponding to each sleep stage. Shape (n_bands, n_stages)
        avg_stage_power_fractions = avg_stage_powers / np.sum(avg_stage_powers, axis=0)

        return avg_stage_power_fractions, power_fractions

    def plot_power_fractions(self, avg_stage_power_fractions):
        # Assuming your avg_power_fractions looks something like this:
        # avg_power_fractions = np.array([[0.1, 0.15, 0.12, 0.18], # Wake delta->beta
        #                                [0.2, 0.25, 0.22, 0.28],
        #                                [0.3, 0.35, 0.32, 0.38],
        #                                [0.4, 0.45, 0.42, 0.48]]) #N3 delta->beta

        # Bands and stages
        bands = ["Delta", "Theta", "Alpha", "Beta"]
        barstages = ["W", "REM", "N1/N2", "N3"]

        # Creating the stacked bar plot
        bar_width = 0.6
        x = np.arange(len(barstages))

        bottoms = np.zeros(len(barstages))

        for i in range(len(bands)):
            plt.bar(x, avg_stage_power_fractions[i], bar_width, label=bands[i], bottom=bottoms,
                    color=['midnightblue', 'magenta', 'orangered', 'lawngreen'][i])
            bottoms += avg_stage_power_fractions[i]

        plt.xlabel('Sleep Stage')
        plt.ylabel('Average Power Fraction')
        plt.title('EEG Example Average Power Distribution')
        plt.xticks(x, barstages)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        plt.show()

    # Save frequency features and labels to csv
    def save_f_features_labels(self, nsrrid, features, epochs):
        # Find the directory to save data to
        data_dir = get_data_dir_shhs(data_type="f", art_rejection=self.params["art_rejection"],
                                     filtering=self.params["lpf"],
                                     prec_epochs=0, foll_epochs=0)
        # Ensure the directory exists
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        # Find the examples that have a label (Boolean Indexing)
        label_bool_idx = np.logical_not(epochs.metadata["Sleep Stage"].isna())
        # Select the features and labels where the two experts agree, using boolean indexing.
        selected_features = features[label_bool_idx, :, :]
        selected_labels = np.array(epochs.metadata["Sleep Stage"])[label_bool_idx]
        # Reshape features to be 2D (examples by features)
        selected_features = selected_features.reshape(selected_features.shape[0], -1)
        # Concatenate features and labels into a single array, with labels as the last column
        data = np.hstack((selected_features, selected_labels[:, np.newaxis]))
        # Save the numpy array
        file_path = data_dir / f"nsrrid_{nsrrid}.npy"
        np.save(file_path, data)
        return

    # Save time domain features and labels to npy
    def save_t_features_labels_npy(self, nsrrid, epochs: mne.Epochs,
                                   incl_preceeding_epochs: int, incl_following_epochs: int):
        # Get directory for processed data, create it if it doesn't exist
        data_dir = get_data_dir_shhs(data_type="t", art_rejection=self.params["art_rejection"], filtering=self.params["lpf"],
                                     prec_epochs=incl_preceeding_epochs, foll_epochs=incl_following_epochs)
        if np.logical_not(os.path.isdir(data_dir)):
            os.makedirs(data_dir)

        # Construct time domain features
        data = epochs.get_data()
        labels = epochs.metadata["Sleep Stage"]
        n_epochs, n_samples_per_epoch = data.shape[0], data.shape[2]
        t_features = []
        t_labels = []

        for i in range(n_epochs):

            # Check that this epoch has a label
            if np.isnan(labels[i]):  # Converting stage list to dataframe converted None -> nan
                continue  # Skip this epoch if it doesn't have a label.

            # Check that the required preceeding and following epochs are present
            start_idx = i - incl_preceeding_epochs
            end_idx = i + incl_following_epochs + 1
            if start_idx < 0 or end_idx > n_epochs:
                continue  # Skip this epoch if too close to edges of recording.

            t_features.append(data[start_idx:end_idx, :, :].flatten())
            t_labels.append(labels[i])

        t_features = np.array(t_features)
        t_labels = np.expand_dims(np.array(t_labels), 1)

        # Save to dataframe
        data = np.concatenate((t_features, t_labels), axis=1)
        data = np.float32(data)     # Float32 takes less space compared to float64
        np.save(file=data_dir/f"nsrrid_{nsrrid}",
                arr=data,
                allow_pickle=False)    # npy is more space and read efficient than csv


if __name__ == "__main__":

    pre = SHHSPreprocessor()
    pre.process_f()