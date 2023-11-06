from datasets import ISRUCDataset
from utils import confusion_matrix
from scripts import TrainMLP
from src.models import MLP1

import torch
from torch.utils.data import DataLoader
import numpy as np


patients = {"train": [1, 2, 3, 4, 5, 6, 7],
            "val":   [8, 9],
            "test":  [10]}
mlp_trainer = TrainMLP(patients=patients)
mlp_trainer.train()
mlp_trainer.save_best_model()
mlp_trainer.evaluate_scores()
mlp_trainer.plot_loss()
