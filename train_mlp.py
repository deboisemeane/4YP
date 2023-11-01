
from src.datasets import ISRUCDataset
from src.models import MLP1
from utils import calculate_ce_loss
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define train, val, test sets.
train_dataset = ISRUCDataset(patients=[1, 2, 3, 4, 5, 6, 7])
val_dataset = ISRUCDataset(patients=[8, 9])
test_dataset = ISRUCDataset(patients=[10])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the model as MLP 20-10-4 i.e. one hidden layer.
model = MLP1()

# Choose criterion and optimiser
lr = 0.05
weight_decay = 0  # L2 penalisation
momentum = 0.1  # Relative weight placed on velocity / accumulated gradient

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

# Initialise lists for plotting loss
TL = []
VL = []


# Training loop
num_epochs = 20
for epoch in range(num_epochs):  # Loops over the entire training set
    running_loss = 0

    # Store calculate and store the current average training and validation losses
    train_loss, train_accuracy = calculate_ce_loss(model, criterion, train_loader)
    val_loss, val_accuracy = calculate_ce_loss(model, criterion, val_loader)
    TL.append(train_loss)
    VL.append(val_loss)

    # Print the current average losses
    print(f"Epoch: [{epoch}/{num_epochs}], Average Training Loss: [{train_loss}]")
    print(f"Epoch: [{epoch + 1}/{num_epochs}], Validation Loss: [{val_loss}], Validation Accuracy: [{val_accuracy}]")

    # Train for one epoch
    for i, batch in enumerate(train_loader):
        x = batch['features']
        labels = batch['label']
        optimiser.zero_grad()   # Because backpropagation accumulates gradients on weights we need to zero them each step.
        y = model(x)
        loss = criterion(y, labels)
        loss.backward()
        optimiser.step()

        # Print Accumulated Training loss every x batches - this is a useful monitoring tool.
        running_loss += loss.item()
        # if (i+1) % 20 == 0:
            # print(f"Epoch: [{epoch+1}/{num_epochs}], Batch: [{i+1}], Average Training Loss This Epoch: [{running_loss / (1+i)}]")

# Store calculate and store the losses after the final epoch
train_loss, train_accuracy = calculate_ce_loss(model, criterion, train_loader)
val_loss, val_accuracy = calculate_ce_loss(model, criterion, val_loader)
TL.append(train_loss)
VL.append(val_loss)

# Plot Training and Validation Loss
x = np.arange(start=1,stop=len(TL))
fig, ax = plt.subplots(1, 1)
ax.plot(TL, 'k', label='Training')
ax.plot(VL, 'r', label='Validation')
ax.set_title(f'Training and Validation Loss for 20-10-4 MLP (lr={lr})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross Entropy Loss')
ax.legend()
plt.show()
