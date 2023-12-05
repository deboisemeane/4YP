import copy
from src.datasets import ISRUCDataset, SHHSDataset
from src.models import MLP1
from utils import calculate_ce_loss, confusion_matrix, accuracy_metrics, get_data_dir_shhs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from pathlib import Path
import random


# Classes used to configure parameters for optimiser instantiation within the TrainMLP class.
class OptimiserConfig:
    def __init__(self, **kwargs):
        self.params = kwargs
        # **kwargs allows us to specify as many hyperparameters as we want.


class AdamConfig(OptimiserConfig):
    def __init__(self, lr=0.001, betas=(0.9, 0.999), **kwargs):
        super().__init__(lr=lr, betas=betas, **kwargs)
        # **kwargs allows us to specify as many hyperparameters as we want.


class SGDConfig(OptimiserConfig):
    def __init__(self, lr=0.05, momentum=0.1, weight_decay=0, **kwargs):
        super().__init__(lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)
        # **kwargs allows us to specify as many hyperparameters as we want.


# DataConfig class contains the parameters required to instantiate a custom dataset.
# Pass as argument to a trainer.
class DataConfig:
    def __init__(self, **kwargs):
        self.params = kwargs


class ISRUCConfig(DataConfig):
    def __init__(self, patients, resample=None, **kwargs):
        super().__init__(patients=patients, resample=resample, data_type="f", **kwargs)
        # patients : dict containing ISRUC patient numbers for "train", "val", "test" datasets.
        # resample : dict containing resample factors for each class "0", "1", "2", "3"
            # !!Resampling will only apply to the training dataset!!


class SHHSConfig(DataConfig):  # This config class is for frequency feature SHHS datasets.
    def __init__(self, split: dict, data_type: str, art_rejection: bool, lpf: bool = True, resample: dict = None, **kwargs):
        super().__init__(resample=resample, data_type=data_type, split=split, art_rejection=art_rejection, lpf=lpf, **kwargs)

        # Choosing between frequency or time domain data.
        data_dir = get_data_dir_shhs(data_type=data_type, art_rejection=art_rejection, lpf=lpf)

        # Read all CSV filenames and extract nsrrids
        all_filenames = os.listdir(data_dir)
        all_nsrrids = [filename.split('.')[0][7::] for filename in all_filenames]

        # Shuffle the list of nsrrids
        random.shuffle(all_nsrrids)

        # Calculate split sizes
        n_train = split["train"]
        n_val = split["val"]
        n_test = split["test"]
        total = n_train + n_val + n_test

        # Check if the total number required exceeds the available number
        if total > len(all_nsrrids):
            raise ValueError(f"Requested total split size ({total}) exceeds the number of available nsrrids ({len(all_nsrrids)}).")

        # Select subsets for each dataset
        train_nsrrids = all_nsrrids[:n_train]
        val_nsrrids = all_nsrrids[n_train:n_train + n_val]
        test_nsrrids = all_nsrrids[n_train + n_val:total]

        # Store the splits
        self.patients = {
            "train": train_nsrrids,
            "val": val_nsrrids,
            "test": test_nsrrids
        }
        self.params.update({"patients": self.patients})


class Train:

    def __init__(self, data_config: DataConfig, optimiser_config: OptimiserConfig,
                 device: torch.device,
                 model=MLP1, ):

        # Set the device
        self.device = device if device is not None else torch.device("cpu")

        # Get dataset parameters from data_config
        self.data_config = data_config
        self.patients = data_config.params["patients"]
        self.resample = data_config.params["resample"]
        self.data_type = data_config.params["data_type"]
        self.art_rejection = data_config.params["art_rejection"]
        self.lpf = data_config.params["lpf"]
        self.data_dir = get_data_dir_shhs(self.data_type, self.art_rejection, self.lpf)

        # Define train, val, test datasets.
        if isinstance(data_config, ISRUCConfig):
            self.train_dataset = ISRUCDataset(patients=self.patients["train"], resample=self.resample)
            self.val_dataset = ISRUCDataset(patients=self.patients["val"])
            self.test_dataset = ISRUCDataset(patients=self.patients["test"])

        elif isinstance(data_config, SHHSConfig):
            self.train_dataset = SHHSDataset(nsrrids=self.patients["train"], data_dir=self.data_dir, resample=self.resample)
            self.val_dataset = SHHSDataset(nsrrids=self.patients["val"], data_dir=self.data_dir)
            self.test_dataset = SHHSDataset(nsrrids=self.patients["test"], data_dir=self.data_dir)

        # Create DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        # Instantiate the model and optimiser.
        self.optimiser_config = optimiser_config
        self.model = model().to(self.device)
        self.optimiser = self.__create_optimiser__(optimiser_config)

        # Initialise other attributes.
        self.best_model_state = None  # Dict that can be loaded to get the model with the lowest validation loss
        self.VL = None  # Training loss after each epoch for the whole training process
        self.TL = None  # Validation loss after each epoch for the whole training process
        self.confusion = None  # Confusion matrix for performance evaluation
        self.metrics = None  # Accuracy metrics for performance evaluation

    def __create_optimiser__(self, config):
        if isinstance(config, SGDConfig):
            optimiser = torch.optim.SGD(self.model.parameters(), **config.params)
        elif isinstance(config, AdamConfig):
            optimiser = torch.optim.Adam(self.model.parameters(), **config.params)
        else:
            raise ValueError("Unsupported optimiser configuration.")
        return optimiser

    def train(self, n_epochs, print_losses=True, weight_losses=True):
        # Display data parameters
        print(f"Data parameters: {self.data_config.params}")
        # Display data split
        print(f"Training with split {self.data_config.params['split']}")
        # Set criterion weight based on inverse of class sizes in the training data.
        weight = self.train_dataset.weight.to(self.device) if weight_losses is True else None
        print(f"Class counts in training data: {self.train_dataset.label_counts}")
        if weight_losses is True:
            print(f"Weighting loss: {weight}")

        # Set criterion and optimiser
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimiser = self.optimiser

        # Initialise lists for plotting loss
        TL = []
        VL = []

        # Initialise best_val_loss which we will use to save the best model state
        best_val_loss = 100000000000

        # Training loop
        for epoch in range(n_epochs):  # Loops over the entire training set
            # Calculate and store the current average training and validation losses
            train_loss, train_accuracy = calculate_ce_loss(self.model, criterion, self.train_loader, self.device)
            val_loss, val_accuracy = calculate_ce_loss(self.model, criterion, self.val_loader, self.device)
            TL.append(train_loss)
            VL.append(val_loss)

            # Check if this is our best performing model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())

            # Print the current average losses
            if print_losses is True:
                print(f"Epoch: [{epoch}/{n_epochs}], Average Training Loss: [{train_loss}]")
                print(f"Epoch: [{epoch}/{n_epochs}], Validation Loss: [{val_loss}], Validation Accuracy: [{val_accuracy}]")

            # Train for one epoch
            for i, batch in enumerate(self.train_loader):
                x = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                optimiser.zero_grad()   # Because backpropagation accumulates gradients on weights we need to zero them each step.
                y = self.model(x)
                loss = criterion(y, labels)
                loss.backward()
                optimiser.step()

        # Calculate and store the losses after the final epoch
        train_loss, train_accuracy = calculate_ce_loss(self.model, criterion, self.train_loader, self.device)
        val_loss, val_accuracy = calculate_ce_loss(self.model, criterion, self.val_loader, self.device)
        TL.append(train_loss)
        VL.append(val_loss)
        self.VL = VL
        self.TL = TL

    def plot_loss(self, ax, labels, t_colour='k', v_colour='r'):
        # Plot Training and Validation Loss
        ax.plot(self.TL, color=t_colour, linestyle='-', label=labels["t"])
        ax.plot(self.VL, color=v_colour, linestyle='--', label=labels["v"])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross Entropy Loss')
        ax.legend(fontsize=8, loc='upper right')

    def save_best_model(self):
        # Save the model state
        if os.path.isdir("model_checkpoints"):
            torch.save(self.best_model_state, f"model_checkpoints/{self.model.__class__.__name__}.pt")
        else:
            os.makedirs("model_checkpoints")
            torch.save(self.best_model_state, f"model_checkpoints/{self.model.__class__.__name__}.pt")


    # Evaluates a confusion matrix on the test dataset for the best model achieved
    def test(self):
        # Load the best model parameters
        self.model.load_state_dict(self.best_model_state)
        # Calculate the confusion matrix
        confusion = confusion_matrix(self.model, self.test_loader, self.device)
        metrics = accuracy_metrics(confusion)
        self.confusion = confusion
        self.metrics = metrics
        print(f"Confusion matrix: \n{confusion}")
        print(f"Accuracy metrics: \n{metrics}")
