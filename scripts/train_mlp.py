import copy
from src.datasets import ISRUCDataset
from src.models import MLP1
from utils import calculate_ce_loss, confusion_matrix
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class TrainMLP:
    def __init__(self,  hyperparameters, patients, model=MLP1, optim=torch.optim.SGD,):
        default_hyperparameters = {"lr": 0.05,          # Currently these are for the SGD optimiser
                                   "weight_decay": 0,  # L2 penalisation
                                   "momentum": 0.1,
                                   "n_epochs": 15}
        self.hyperparameters = default_hyperparameters.update(hyperparameters)

        # Define train, val, test sets.
        self.train_dataset = ISRUCDataset(patients=[1, 2, 3, 4, 5, 6, 7])
        self.val_dataset = ISRUCDataset(patients=[8, 9])
        self.test_dataset = ISRUCDataset(patients=[10])

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        # Define the model as MLP 20-10-4 i.e. one hidden layer, and other attributes of the training algorithm.
        self.model = model()
        self.optim = optim

        # Initialise other attributes
        self.best_model_state = None  # Dict that can be loaded to get the model with the lowest validation loss
        self.VL = None  # Training loss after each epoch for the whole training process
        self.TL = None  # Validation loss after each epoch for the whole training process

        # Initialise axes for plots
        self.fig, self.ax = plt.subplots(1, 1)

    def train(self):

        # Choose criterion and optimiser
        lr = self.hyperparameters["lr"]
        weight_decay = self.hyperparameters["weight_decay"]  # L2 penalisation
        momentum = self.hyperparameters["momentum"]  # Relative weight placed on velocity / accumulated gradient
        n_epochs = self.hyperparameters["n_epochs"]

        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

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
        ax.plot(self.TL, colour=train_colour, label='Training')
        ax.plot(self.VL, colour=val_colour, label='Validation')
        ax.set_title(f'CrossEntropyLoss for MLP (lr={self.hyperparameters["lr"]}, '
                     f'momentum={self.hyperparameters["momentum"]})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross Entropy Loss')
        ax.legend()
        plt.show()

    def save_best_model(self, path="model_checkpoints/MLP1_save.pt"):
        # Save the model state
        torch.save(self.best_model_state, path)

    # Evaluates a confusion matrix on the test dataset for the best model achieved
    def evaluate_scores(self):
        # Load the best model parameters
        self.model.load_state_dict(self.best_model_state)
        # Calculate the confusion matrix
        confusion = confusion_matrix(self.model, self.test_loader)
        print(f"Confusion matrix: {confusion}")
        print(f"Accuracy: {np.trace(confusion) / np.sum(confusion)}")

