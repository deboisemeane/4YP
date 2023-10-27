import mne
import numpy as np
import pandas as pd


class ISRUCPreprocessor:
    def __init__(self, params): #params = {patient, channel}
        self.params = params
        self.epoch, self.stage = self.load_hypnogram()
        self.hr = self.load_heart_rate()
        self.raw_eeg = mne.io.read_raw_edf(f'data/Raw/ISRUC/SubgroupIII/{self.params["patient"]}.edf')
        self.epochs = self.create_epochs()
        self.lpf_epochs = self.lpf_epochs()
        self.freqs, self.linearised_fft = self.apply_hamming_fft()


    def load_hypnogram(self): #Returns [epoch_number, sleep stage]
        y_1 = np.loadtxt(f'data/Raw/ISRUC/SubgroupIII/{self.params["patient"]}_1.txt', dtype=int)
        y_2 = np.loadtxt(f'data/Raw/ISRUC/SubgroupIII/{self.params["patient"]}_2.txt', dtype=int)

        x = np.arange(0, len(y_1))

        y = np.zeros(y_1.shape)                 #Initialise vector of stage labels
        for i in range(x):                      #Creates a vector of stage labels
            if (y_1[i] == 3) and (y_2[i] == 3):
                y[i] = 0                        #   Set 0 - N3
            elif ((y_1[i]==1)or(y_1[i]==2)) and ((y_1[i]==1)or(y_2[i]==2)):
                y[i] = 1                        #   Set 1 - N1,N2
            elif (y_1[i]==5) and (y_2[i]==5):
                y[i] = 2                        #   Set 2 - REM
            elif (y_1[i]==0) and (y_2[i]==0):
                y[i] = 3                        #   Set 3 - W
            else:
                y[i] = None                     #   Set None - experts do not agree

        epoch, stage = x, y

        return epoch, stage

    def load_heart_rate(self):  #Returns [HR]
        xl = pd.read_excel(f'data/Raw/ISRUC/SubgroupIII/{self.params["patient"]}_1.txt')
        y = np.array(xl['HR'])

        # Remove typos / non integer entries
        for i in range(len(y)):
            if type(y[i]) != int:
                y[i] = None
        #Remove artificial peaks >120bpm
            if y[i] > 120:
                y[i] = None
        hr = y
        return hr

    def create_epochs(self):    #Creates mne.Epochs object
                                # - consists of the 30s windows with metadata labels
                                # - labels=None for epochs where experts do not agree
        return

    def lpf_epochs(self):

        return

    def apply_hamming_fft(self):

        return

    def select_freq_features(self):

        return
