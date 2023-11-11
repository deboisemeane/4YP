from scripts import TrainMLP, AdamConfig, SGDConfig, ISRUCConfig
from src.models import MLP1
import matplotlib.pyplot as plt


# Investigating the effect of Resampling REM
patients = {"train": [1, 2, 3, 4, 5, 6, 7],
            "val": [8, 9],
            "test": [10]}
resample = {"2": 5}  # Resample REM by a chosen factor.

data_config = ISRUCConfig(patients=patients, resample=resample)
optimiser_config = AdamConfig()

trainer = TrainMLP(data_config=data_config, optimiser_config=optimiser_config, model=MLP1)

trainer.train(n_epochs=1, print_losses=True)

trainer.evaluate_accuracy()
confusion = trainer.confusion

df = trainer.train_dataset.data
x = []
for i in range(4):
    stage_data = df[df.iloc[:, 21].astype(int) == i]
    x.append(len(stage_data))

print(df.head())
stage_proportions = [a/sum(x) for a in x]
print(stage_proportions)
