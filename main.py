import torch
from scripts import Train, AdamConfig, SHHSConfig, KFoldConfig, KFold_CV
from src.models import MLP1, Sors, Sors7, Sors_nocontext1, Sors_nocontext2
from debug import AFNet
import matplotlib.pyplot as plt
from utils import Timer
import numpy as np


def main():
    # Find device
    print("Using cuda" if torch.cuda.is_available() else "Using cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training the MLP on SHHS data
    #split = {"train": 10, "val": 0, "test": 0}
    split = {"train": 1950, "val": 557, "test": 278}  # EEG 7-2-1 split
    #split = {"train": 1926, "val": 550, "test": 275}  # CARDIO 7-2-1 split
    #split = {"train": 350, "val": 100, "test": 50}
    #resample = {"2": 2.84}

    data_config = KFoldConfig(k=5, split=split, data_type="t", art_rejection=True, filtering=True, resample=None,
                             prec_epochs=2, foll_epochs=1)
    optimiser_config = AdamConfig(lr=0.0003)


    kfoldcv = KFold_CV(data_config=data_config, optimiser_config=AdamConfig(lr=0.0003), device=device, model=Sors)

    # Generate random loss data for testing the plot
    # Assuming there are 5 folds (k=5) and 12 epochs

    num_folds = 5
    num_epochs = 12

    # Random training and validation losses for each fold and each epoch
    # Generating losses between 0.0 and 1.0 as an example
    kfoldcv.TLs = [np.random.rand(num_epochs) for _ in range(num_folds)]
    kfoldcv.VLs = [np.random.rand(num_epochs) for _ in range(num_folds)]
    kfoldcv.plot_loss("figures/5foldcv_t.png", title="5-Fold Cross Validation Loss Plot, Sors EEG")


if __name__ == '__main__':
    main()
