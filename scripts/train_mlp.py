import copy
from src.datasets import ISRUCDataset
from src.models import MLP1
from utils import calculate_ce_loss, confusion_matrix, accuracy_metrics
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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


class TrainMLP:
    # Attributes shared by all instances go outside any methods.
    fig, ax = None, None

    def __init__(self, patients, optimiser_config, model=MLP1):

        # Define train, val, test sets.
        self.train_dataset = ISRUCDataset(patients=patients["train"])
        self.val_dataset = ISRUCDataset(patients=patients["val"])
        self.test_dataset = ISRUCDataset(patients=patients["test"])

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        # Instantiate the model and optimiser.
        self.optimiser_config = optimiser_config
        self.model = model()
        self.optimiser = self.create_optimiser(optimiser_config)

        # Initialise other attributes.
        self.best_model_state = None  # Dict that can be loaded to get the model with the lowest validation loss
        self.VL = None  # Training loss after each epoch for the whole training process
        self.TL = None  # Validation loss after each epoch for the whole training process

        # Initialise axes for plots (attributes shared by all instances of TrainMLP)
        if TrainMLP.fig is None or TrainMLP.ax is None:
            TrainMLP.fig, self.ax = plt.subplots(1, 1)

    def create_optimiser(self, config):
        if isinstance(config, SGDConfig):
            optimiser = torch.optim.SGD(self.model.parameters(), **config.params)
        elif isinstance(config, AdamConfig):
            optimiser = torch.optim.Adam(self.model.parameters(), **config.params)
        else:
            raise ValueError("Unsupported optimiser configuration.")
        return optimiser

    def train(self, n_epochs):

        # Set criterion and optimiser
        criterion = nn.CrossEntropyLoss()
        optimiser = self.optimiser

        # Initialise lists for plotting loss
        TL = []
        VL = []

        # Initialise best_val_loss which we will use to save the best model state
        best_val_loss = 100000000000

        # Training loop
        for epoch in range(n_epochs):  # Loops over the entire training set
            # Calculate and store the current average training and validation losses
            train_loss, train_accuracy = calculate_ce_loss(self.model, criterion, self.train_loader)
            val_loss, val_accuracy = calculate_ce_loss(self.model, criterion, self.val_loader)
            TL.append(train_loss)
            VL.append(val_loss)

            # Check if this is our best performing model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())

            # Print the current average losses
            print(f"Epoch: [{epoch}/{n_epochs}], Average Training Loss: [{train_loss}]")
            print(f"Epoch: [{epoch}/{n_epochs}], Validation Loss: [{val_loss}], Validation Accuracy: [{val_accuracy}]")

            # Train for one epoch
            for i, batch in enumerate(self.train_loader):
                x = batch['features']
                labels = batch['label']
                optimiser.zero_grad()   # Because backpropagation accumulates gradients on weights we need to zero them each step.
                y = self.model(x)
                loss = criterion(y, labels)
                loss.backward()
                optimiser.step()

        # Calculate and store the losses after the final epoch
        train_loss, train_accuracy = calculate_ce_loss(self.model, criterion, self.train_loader)
        val_loss, val_accuracy = calculate_ce_loss(self.model, criterion, self.val_loader)
        TL.append(train_loss)
        VL.append(val_loss)
        self.VL = VL
        self.TL = TL

    def plot_loss(self, val_colour='r', train_colour='k'):
        # Plot Training and Validation Loss
        fig, ax = self.fig, self.ax
        ax.plot(self.TL, color=train_colour, label='Training')
        ax.plot(self.VL, color=val_colour, label='Validation')
        ax.set_title(f'CrossEntropyLoss for MLP ({self.optimiser_config.params})', fontsize=7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross Entropy Loss')
        ax.legend()
        plt.show()

    def save_best_model(self, name="MLP1_save.pt"):
        # Save the model state
        torch.save(self.best_model_state, f"model_checkpoints/{name}")

    # Evaluates a confusion matrix on the test dataset for the best model achieved
    def evaluate_accuracy(self):
        # Load the best model parameters
        self.model.load_state_dict(self.best_model_state)
        # Calculate the confusion matrix
        confusion = confusion_matrix(self.model, self.test_loader)
        metrics = accuracy_metrics(confusion)
        print(f"Confusion matrix: \n{confusion}")
        print(f"Accuracy: {metrics['ACC']}")
        print(f"Sensitivity: {metrics['TPR']}")
        print(f"Specificity: {metrics['TNR']}")
