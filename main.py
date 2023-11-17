from scripts import TrainMLP, AdamConfig, SGDConfig, ISRUCConfig, SHHSConfig_f
from utils import accuracy_metrics
from visualisation import plot_norm_confusion
from src.models import MLP1
import matplotlib.pyplot as plt


# Training the MLP on SHHS data

split = {"train": 350,
         "val":   150,
         "test":  50}

data_config = SHHSConfig_f(split=split)
optimiser_config = AdamConfig(lr=0.00005)

trainer = TrainMLP(data_config=data_config, optimiser_config=optimiser_config, model=MLP1)
trainer.train(n_epochs=50, print_losses=True)
trainer.save_best_model()

# Plotting loss for training with SHHS
fig, ax = plt.subplots()
ax.set_title("CrossEntropyLoss training 20-10-4 MLP with SHHS-1")
labels = {"t": "Training",
          "v": "Validation"}
trainer.plot_loss(ax=ax, labels=labels)
plt.show()

trainer.test()
plot_norm_confusion(trainer.confusion, 'MLP trained on SHHS-1 dataset (350-150-50 split)')
