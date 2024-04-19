import torch
from scripts import Train, AdamConfig, SHHSConfig, KFoldConfig, KFold_CV
from src.models import MLP1, Sors, Sors7, Sors_nocontext1, Sors_nocontext2, Sors_dualcnn, Sors_largekernels, Sors_rip_hr
import matplotlib.pyplot as plt
from utils import Timer


def main():
    # Find device
    print("Using cuda" if torch.cuda.is_available() else "Using cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training the MLP on SHHS data
    #split = {"train": 15, "val": 10, "test": 2}
    #split = {"train": 1950, "val": 557, "test": 278}
    #split = {"train": 1926, "val": 550, "test": 275}
    #split = {"train": 350, "val": 100, "test": 50}
    #resample = {"2": 2.84}
    split = {"train": 0.7, "val": 0.2, "test": 0.1}

    data_config = KFoldConfig(k=5, split=split, data_type="rip_hr", art_rejection=True, filtering=True, resample=None,
                             prec_epochs=2, foll_epochs=1, equal_split="val")
    optimiser_config = AdamConfig(lr=0.0003)

    #trainer = Train(data_config=data_config, optimiser_config=optimiser_config, model=Sors, device=device)
    kfoldcv = KFold_CV(data_config, optimiser_config, device, Sors_rip_hr)
    timer = Timer()
    timer.start()
    kfoldcv.cross_validate(n_epochs=12, print_losses=True, weight_losses=True, weight_scalings=torch.tensor([1, 1.5, 1, 1]))
    time_train = timer.stop()
    print(f"Total training time: {time_train}")

    # Plotting loss for training with SHHS
    kfoldcv.plot_loss("/figures/5fold_Sorsdualstream_rip_hr.png", "5-Fold Cross Validation of RIP HR Dual Stream Model")



if __name__ == '__main__':
    main()
