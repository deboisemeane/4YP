from scripts import TrainMLP, AdamConfig, SGDConfig, ISRUCConfig
from src.models import MLP1


patients = {"train": [1, 2, 3, 4, 5, 6, 7],
            "val":   [8, 9],
            "test":  [10]}

# We want to resample REM
resample = {"2": 2}


data_config = ISRUCConfig(patients=patients, resample=resample)

optimiser_config = AdamConfig()

mlp_trainer = TrainMLP(data_config=data_config, model=MLP1, optimiser_config=optimiser_config)
mlp_trainer.train(n_epochs=25)
mlp_trainer.save_best_model()
mlp_trainer.evaluate_accuracy()
mlp_trainer.plot_loss()
