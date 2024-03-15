import torch
from scripts import Train, AdamConfig, SHHSConfig
from src.models import MLP1, Sors, Sors7, Sors_nocontext1, Sors_nocontext2
from debug import AFNet
import matplotlib.pyplot as plt
from utils import Timer


def main():
    # Find device
    print("Using cuda" if torch.cuda.is_available() else "Using cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training the MLP on SHHS data
    #split = {"train": 10, "val": 0, "test": 0}
    split = {"train": 1950, "val": 557, "test": 278}  # EEG 7-2-1 split
    #split = {"train": 1926, "val": 550, "test": 275}  # CARDIO 7-2-1 split
    #split = {"train": 350, "val": 100, "test": 50}
    #resample = {"2": 2.84}

    data_config = SHHSConfig(split=split, data_type="t", art_rejection=True, filtering=True, resample=None,
                             prec_epochs=2, foll_epochs=1)
    optimiser_config = AdamConfig(lr=0.001)

    trainer = Train(data_config=data_config, optimiser_config=optimiser_config, model=Sors, device=device)

    timer = Timer()
    timer.start()
    trainer.train(n_epochs=12, print_losses=True, weight_losses=True)
    time_train = timer.stop()
    print(f"Total training time: {time_train}")

    # Testing
    trainer.test()

    # Plotting loss for training with SHHS
    fig, ax = plt.subplots()
    ax.set_title("EEG Experiment 1")
    labels = {"t": "Training", "v": "Validation"}
    trainer.plot_loss(ax=ax, labels=labels)
    plt.savefig(f'figures/eeg_experiment_1_{split["train"]}-{split["val"]}-{split["test"]}.png')


if __name__ == '__main__':
    main()
