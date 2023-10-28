import mne
import numpy as np
import pandas as pd


class ISRUCPreprocessor:
    def __init__(self, **params):  # ** Allows us to specify any number of the parameters, which are collected to dict

        default_params = {"patient": 7,
                          "channel": "C4-A1",
                          "lpf_cutoff": 30,
                          "feature_bin_freqwidth": 1,
                          "n_features": 20
                          }
        self.params = default_params
        self.params.update(params)
        self.time, self.stage = self.load_hypnogram()
        self.heart_rate = self.load_heart_rate()
        self.raw_eeg = self.load_raw_eeg()
        self.epochs = self.create_epochs()
        self.lpf_epochs = self.lpf_epochs()
        self.freqs, self.fft_data = self.apply_hamming_fft()
        self.feature_freqs, self.features = self.select_freq_features()

    def load_raw_eeg(self):  # Returns mne.Raw for chosen channel
        raw_eeg = mne.io.read_raw_edf(
            f'data/Raw/ISRUC/SubgroupIII/{self.params["patient"]}/{self.params["patient"]}.edf')
        raw_eeg = raw_eeg.pick_channels([self.params["channel"]])
        return raw_eeg

    def load_hypnogram(self):  # Returns [epoch_number, sleep stage]
        y_1 = np.loadtxt(f'data/Raw/ISRUC/SubgroupIII/{self.params["patient"]}/{self.params["patient"]}_1.txt',
                         dtype=int)
        y_2 = np.loadtxt(f'data/Raw/ISRUC/SubgroupIII/{self.params["patient"]}/{self.params["patient"]}_2.txt',
                         dtype=int)

        x = np.arange(0, len(y_1))  # Time in epochs

        y = np.zeros(y_1.shape)  # Initialise vector of stage labels
        for i in range(len(y_1)):  # Creates a vector of stage labels
            if (y_1[i] == 3) and (y_2[i] == 3):
                y[i] = 0  # Set 0 - N3
            elif ((y_1[i] == 1) or (y_1[i] == 2)) and ((y_1[i] == 1) or (y_2[i] == 2)):
                y[i] = 1  # Set 1 - N1,N2
            elif (y_1[i] == 5) and (y_2[i] == 5):
                y[i] = 2  # Set 2 - REM
            elif (y_1[i] == 0) and (y_2[i] == 0):
                y[i] = 3  # Set 3 - W
            else:
                y[i] = None  # Set None - experts do not agree

        time, stage = x, y

        return time, stage

    def load_heart_rate(self):  # Returns [HR]
        xl = pd.read_excel(f'data/Raw/ISRUC/SubgroupIII/{self.params["patient"]}/{self.params["patient"]}_1.xlsx')
        y = np.array(xl['HR'])

        # Remove typos / non integer entries
        for i in range(len(y)):
            if type(y[i]) is not int:
                y[i] = None
            # Remove artificial peaks >120bpm
            elif y[i] > 120:
                y[i] = None
        heart_rate = y
        return heart_rate

    def create_epochs(self):  # Creates mne.Epochs object
        detrend = 0  # D.c. de-trend of eeg signal around 0
        metadata = pd.DataFrame({"Sleep Stage": self.stage})  # Reminder: stage = None when experts disagree
        events = mne.make_fixed_length_events(self.raw_eeg, duration=30)  # 30s windows
        epochs = mne.Epochs(self.raw_eeg, events, tmin=0, tmax=30, preload=True,
                            detrend=detrend, metadata=metadata, baseline=None, )  # Preload = True for filtering
        return epochs

    def lpf_epochs(self):  # IIR filter to prevent
        params = self.params
        lpf_epochs = self.epochs.copy().filter(l_freq=None, h_freq=params["lpf_cutoff"], method='fir',
                                               fir_design='firwin', phase='zero')

        return lpf_epochs

    def apply_hamming_fft(self):  # Returns frequencies and corresponding absolute DFT components
        # Apply hamming window - reduces spectral leakage
        window_size = self.lpf_epochs.get_data().shape[2]  # Size = no. of points in each epoch
        hamming_window = np.hamming(window_size)

        # Apply FFT
        lpf_epochs = self.lpf_epochs
        windowed_data = lpf_epochs.get_data() * hamming_window
        fft_data = np.abs(np.fft.rfft(windowed_data, axis=2))  # Shape (n_epochs, n_channels(1), epoch_length)

        # Create vector of corresponding frequencies
        sfreq = lpf_epochs.info["sfreq"]
        nyquist_freq = sfreq / 2  # Maximum frequency in the FFT
        endpoint = fft_data.shape[2] % 2 == 0   # If number of points is even, the highest frequency is fs/2 (endpoint=True)
        freqs = np.linspace(0, nyquist_freq, len(fft_data[0][0]), endpoint=endpoint)  # Evenly spaced from 0 to nyquist freq

        return freqs, fft_data

    def select_freq_features(self):  # Averages fft data over specified frequency bins

        freqs = self.freqs
        fft_data = self.fft_data
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
        epoch_energies = np.sum(binned_data ** 2, axis=0)  # Summing over features - energy for each epoch
        normalised_features = binned_data / np.sqrt(epoch_energies)

        return feature_freqs, normalised_features

    def save_features_labels_csv(self):

        return
