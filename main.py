import torch.cuda

from scripts import Train, AdamConfig, SGDConfig, ISRUCConfig, SHHSConfig
from utils import accuracy_metrics
from visualisation import plot_norm_confusion
from src.models import MLP1, MLP2
import matplotlib.pyplot as plt

# Find device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training the MLP on SHHS data

split = {"train": 2000,
         "val":   580,
         "test":  290}

resample = {"2": 2.84}

data_config = SHHSConfig(split=split, data_type="f", resample=None)
optimiser_config = AdamConfig(lr=0.0001)

trainer = Train(data_config=data_config, optimiser_config=optimiser_config, model=MLP1, device=device)
trainer.train(n_epochs=30, print_losses=True, weight_losses=True)

trainer.save_best_model()

# Plotting loss for training with SHHS
fig, ax = plt.subplots()
ax.set_title("Weighted CrossEntropyLoss training 20-10-4 MLP with SHHS-1")
labels = {"t": "Training",
          "v": "Validation"}
trainer.plot_loss(ax=ax, labels=labels)
plt.savefig(f'figures/weighted_loss_MLP1_shhs1_{split["train"]}-{split["val"]}-{split["test"]}.png')

# Testing
trainer.test()
plot_norm_confusion(trainer.confusion, 'MLP trained on SHHS-1 dataset (350-150-50 split)')
