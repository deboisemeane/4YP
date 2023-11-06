from scripts import TrainMLP, AdamConfig, SGDConfig
from src.models import MLP1


patients = {"train": [1, 2, 3, 4, 5, 6, 7],
            "val":   [8, 9],
            "test":  [10]}

optimiser_config = SGDConfig(lr=0.05, momentum=0.5)
mlp_trainer = TrainMLP(patients=patients, model=MLP1, optimiser_config=optimiser_config)
mlp_trainer.train(n_epochs=50)
mlp_trainer.save_best_model()
mlp_trainer.evaluate_accuracy()
mlp_trainer.plot_loss()
