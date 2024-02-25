from preprocess import SHHSCardioPreprocessor
import torch
pre = SHHSCardioPreprocessor()
pre.process(data_types=["THOR RES"])


