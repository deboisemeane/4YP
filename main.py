from datasets import ISRUCDataset
from utils import confusion_matrix
from src.models import MLP1
import torch
from torch.utils.data import DataLoader
import numpy as np

test_dataset = ISRUCDataset([10])

model_state = torch.load("model_checkpoints/MLP1_save.pt")
model = MLP1()
model.load_state_dict(model_state)
model.eval()

dataloader = DataLoader(test_dataset, batch_size=32)

confusion = confusion_matrix(model, dataloader)
accuracy = np.trace(confusion) / np.sum(confusion)

print(confusion)
print(f"Accuracy: {accuracy}")