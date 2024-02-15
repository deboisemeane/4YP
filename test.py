from preprocess import SHHSCardioPreprocessor
import torch
pre = SHHSCardioPreprocessor()
pre.process(data_types=["THOR RES"])
pre.process(data_types=["ECG"])
pre.process(data_types=["THOR RES, ECG"])


