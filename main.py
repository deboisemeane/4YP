import torch
from scripts import Train, AdamConfig, SHHSConfig
from src.models import MLP1, Sors
import matplotlib.pyplot as plt
from utils import Timer


def main():
    # Find device
    print("Using cuda" if torch.cuda.is_available() else "Using cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda")

    # Training the MLP on SHHS data
    #split = {"train": 15, "val": 10, "test": 2}
    split = {"train": 1950, "val": 557, "test": 278}
    #split = {"train": 350, "val": 100, "test": 50}
    #resample = {"2": 2.84}

    data_config = SHHSConfig(split=split, data_type="t", art_rejection=True, lpf=True, resample=None)
    optimiser_config = AdamConfig(lr=0.0001)

    trainer = Train(data_config=data_config, optimiser_config=optimiser_config, model=Sors, device=device)

    timer = Timer()
    timer.start()
    trainer.train(n_epochs=10, print_losses=True, weight_losses=False)
    time_train = timer.stop()
    print(f"Total training time: {time_train}")

    # Testing
    trainer.test()

    # Plotting loss for training with SHHS
    fig, ax = plt.subplots()
    ax.set_title("Training CNN-12")
    labels = {"t": "Training", "v": "Validation"}
    trainer.plot_loss(ax=ax, labels=labels)
    plt.savefig(f'figures/artrejection1_lpf1_weighted0_CNN12_shhs1_{split["train"]}-{split["val"]}-{split["test"]}.png')


if __name__ == '__main__':
    main()
