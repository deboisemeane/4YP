from preprocess import SHHSCardioPreprocessor
import torch
pre = SHHSCardioPreprocessor()
pre.process(data_types=["ECG", "THOR RES"], incl_preceeding_epochs=2, incl_following_epochs=1)


