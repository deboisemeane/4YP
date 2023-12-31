import torch
import numpy as np


def confusion_matrix(model, dataloader, device):   # 4 Class confusion matrix
    model.eval()
    with torch.no_grad():
        confusion = np.zeros((4, 4))
        for i, batch in enumerate(dataloader):
            x = batch["features"].to(device)
            labels = batch["label"].to(device)
            predictions = model(x).argmax(dim=1)
            for actual in range(4):                 # Rows = Actual (0:N3 -> 3:W)
                for predicted in range(4):          # Columns = Predicted (0:N3 -> 3:W)
                    confusion[actual, predicted] += torch.sum((labels == actual) & (predictions == predicted)).item()

        return confusion
