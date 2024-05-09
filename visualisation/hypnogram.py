import torch
import numpy as np
from utils import get_data_dir_shhs
import matplotlib.pyplot as plt
from pathlib import Path
from src.models import Sors, Sors_rip_hr, MLP1


def predict_hypnogram(model, nsrrid, data_type, art_rejection=True, filtering=True, prec_epochs=2, foll_epochs=1):
    data_dir = get_data_dir_shhs(data_type=data_type, art_rejection=art_rejection, filtering=filtering,
                                 prec_epochs=prec_epochs, foll_epochs=foll_epochs)
    data = np.load(data_dir / f"nsrrid_{nsrrid}.npy").squeeze()
    root_dir = Path(__file__).parent.parent

    x = torch.tensor(data[:, :-1], dtype=torch.float32).unsqueeze(1)  # Unsqueeze to get channel dimension
    labels = torch.tensor(data[:, -1], dtype=torch.long)
    model_state_dict = torch.load(root_dir / f"model_checkpoints/{model.__name__}.pt", map_location=torch.device('cpu'))
    model = model()
    model.load_state_dict(model_state_dict)
    model.eval()
    ground_truth = []
    predictions = []
    for i in range(x.shape[0]):
        scores = model(x[i, :, :].unsqueeze(0))
        predictions.append(torch.argmax(scores))
        ground_truth.append(labels[i])
    return labels, predictions


if __name__ == "__main__":
    fs = 12
    nsrrid = 201463  # 201979, 201463
    labels, predictions = predict_hypnogram(Sors, nsrrid, "t")
    fig, ax = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(12,4)
    ax[0].plot(np.arange(len(labels)) / 120, labels, color="k",)
    ax[1].plot(np.arange(len(labels)) / 120, predictions, color='dodgerblue')
    labels, predictions = predict_hypnogram(Sors_rip_hr, nsrrid, "rip_hr")
    ax[2].plot(np.arange(len(labels)) / 120, predictions, color='red')
    ax[2].set_xlabel("Time (hrs)", fontsize=fs)

    for ax in ax:
        yticklabels = ['N3', 'N1/N2', 'REM', 'W']
        yticks = [0, 1, 2, 3]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    fig.text(0.04, 0.5, 'Sleep Stage', va='center', rotation='vertical', fontsize=fs)
    fig.text(0.93, 0.5, '    EEG\nACC: 0.89', va='center', rotation=270, fontsize=fs)
    fig.text(0.93, 0.77, 'Target', va='center', rotation=270, fontsize=fs)
    fig.text(0.93, 0.22, '  RIP, HR\nACC: 0.68', va='center', rotation=270, fontsize=fs)
    fig.suptitle("Male, 75", x=0.5, y=0.95)
    plt.show()

