import os
import numpy as np
import torch
from scripts import Train
from scripts.train import DataConfig, OptimiserConfig, AdamConfig
from src.models import Sors
from utils import get_data_dir_shhs
import matplotlib.pyplot as plt
from pathlib import Path


class KFoldConfig(DataConfig):

    def __init__(self, k: int, split: dict, data_type: str, prec_epochs: int, foll_epochs: int, art_rejection: bool = True,
                 filtering: bool = True, resample: dict = None, **kwargs):
        """

        :param k: Number of folds for cross validation.
        :param split: dict containing proportions of total data for train, test. n_val is determined by n_total / k
        :param data_type: string determining signal(s) to be used.
        :param prec_epochs: context - number of preceeding 30s epochs.
        :param foll_epochs: context - number of following 30s epochs.
        :param art_rejection: bool indicating artefact rejection.
        :param filtering: bool indicating filtering.
        :param resample: dict indicating resampling.
        :param kwargs:
        """
        super().__init__(k=k, resample=resample, data_type=data_type, split=split, art_rejection=art_rejection,
                         lpf=filtering,
                         prec_epochs=prec_epochs, foll_epochs=foll_epochs, **kwargs)

        # Get processed recordings that meet our desired preprocessing.
        data_dir = get_data_dir_shhs(data_type=data_type, art_rejection=art_rejection, filtering=filtering,
                                     prec_epochs=prec_epochs, foll_epochs=foll_epochs)
        # Read all filenames and extract nsrrids
        all_filenames = os.listdir(data_dir)
        all_nsrrids = [filename.split('.')[0][7::] for filename in all_filenames]
        n_total = len(all_nsrrids)

        # Generate the splits
        self.all_nsrrids = all_nsrrids
        self.splits = []

        np.random.shuffle(self.all_nsrrids)  # Shuffle the data to ensure randomness

        segment_length = len(self.all_nsrrids) // k
        for fold in range(k):
            start_val = fold * segment_length
            end_val = start_val + segment_length

            val_nsrrids = self.all_nsrrids[start_val:end_val]

            train_test_nsrrids = self.all_nsrrids[:start_val] + self.all_nsrrids[end_val:]
            n_train = int(np.floor(len(train_test_nsrrids) * split["train"]))

            train_nsrrids = train_test_nsrrids[:n_train]
            test_nsrrids = train_test_nsrrids[n_train:]

            patients = {"train": train_nsrrids, "val": val_nsrrids, "test": test_nsrrids}
            self.splits.append(patients)

    def set_patients(self, fold):
        self.params.update({"patients": self.splits[fold]})


class KFold_CV:

    def __init__(self, data_config: KFoldConfig, optimiser_config: OptimiserConfig,
                 device: torch.device, model):
        # Training objects
        self.data_config = data_config
        self.optimiser_config = optimiser_config
        self.device = device
        self.model = model
        self.trainer = None
        # Inference objects
        self.TLs = []
        self.VLs = []
        self.ACCs = []
        self.Kappas = []
        self.confusion = np.zeros((4, 4))  # Total confusion matrix across all tests.
        self.best_accuracy = 0
        self.best_model_state = None # Save this for
        self.best_fold = 0

    def cross_validate(self, n_epochs: int, print_losses=True, weight_losses=True):

        # Iterate over folds
        for fold in range(self.data_config.params["k"]):
            print(f"Training fold {fold+1}")
            # Iterate training split
            self.data_config.set_patients(fold)
            # Reset Trainer
            self.trainer = Train(data_config=self.data_config, optimiser_config=self.optimiser_config,
                                 device=self.device, model=self.model)
            # Train
            self.trainer.train(n_epochs=n_epochs, print_losses=print_losses, weight_losses=weight_losses)
            self.TLs.append(self.trainer.TL)
            self.VLs.append(self.trainer.VL)
            # Test
            self.trainer.test()
            self.confusion += self.trainer.confusion
            self.ACCs.append(self.trainer.metrics["Total ACC"])
            self.Kappas.append(self.trainer.metrics["Total Kappa"])
            if self.trainer.metrics["Total ACC"] > self.best_accuracy:
                self.best_accuracy = self.trainer.metrics["Total ACC"]
                self.best_model_state = self.trainer.best_model_state
                self.best_fold = fold
        # Save the best model overall
        self.trainer.best_model_state = self.best_model_state
        self.trainer.save_best_model()
        # Print Results
        print(f"TLs: {self.TLs}")
        print(f"VLs: {self.VLs}")
        print(f"Accuracies: {self.ACCs}")
        print(f"Kappas: {self.Kappas}")
        print(f"Best fold: {self.best_fold+1}")

    def plot_loss(self, figure_dir: str, title: str):
        fig, ax = plt.subplots()

        # Convert lists of losses to numpy arrays for easier manipulation
        TLs_array = np.array(self.TLs)
        VLs_array = np.array(self.VLs)

        # Calculate median and IQR for training and validation losses
        TL_median = np.median(TLs_array, axis=0)
        VL_median = np.median(VLs_array, axis=0)
        TL_iqr = np.subtract(*np.percentile(TLs_array, [75, 25], axis=0))
        VL_iqr = np.subtract(*np.percentile(VLs_array, [75, 25], axis=0))

        epochs = np.arange(len(TL_median))

        # Plot training losses
        ax.plot(epochs, TL_median, label='Training Loss', color='darkblue')
        ax.fill_between(epochs, TL_median - 0.5 * TL_iqr, TL_median + 0.5 * TL_iqr, alpha=0.2, color='darkblue', label="Training Loss IQR")

        # Plot validation losses
        ax.plot(epochs, VL_median, label='Validation Loss', color='gold')
        ax.fill_between(epochs, VL_median - 0.5 * VL_iqr, VL_median + 0.5 * VL_iqr, alpha=0.2, color='gold', label="Validation Loss IQR")

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Cross Entropy Loss')
        ax.set_title(title)
        ax.legend()

        plt.savefig(figure_dir)


if __name__ == "__main__":
    k = 5
    split = {"train": 0.7, "val": 0.2, "test": 0.1}
    data_type = "t"
    prec_epochs = 2
    foll_epochs = 1

    kfold = KFoldConfig(k=k, data_type=data_type, split=split, prec_epochs=prec_epochs, foll_epochs=foll_epochs)
    kfoldcv = KFold_CV(data_config=kfold, optimiser_config=AdamConfig(lr=0.0003), device=torch.device("cpu"), model=Sors)

    # Generate random loss data for testing the plot
    # Assuming there are 5 folds (k=5) and 12 epochs

    num_folds = 5
    num_epochs = 12

    # Random training and validation losses for each fold and each epoch
    # Generating losses between 0.0 and 1.0 as an example
    kfoldcv.TLs = [np.random.rand(num_epochs) for _ in range(num_folds)]
    kfoldcv.VLs = [np.random.rand(num_epochs) for _ in range(num_folds)]
    kfoldcv.plot_loss()
