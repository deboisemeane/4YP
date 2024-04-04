import torch
from scripts import Train, AdamConfig, SHHSConfig
from src.models import MLP1, Sors
import matplotlib.pyplot as plt
from utils import Timer, get_data_dir_shhs
from debug import AFNet, AFNet_wip
from scripts.train import SHHSConfig


def main():
    print(get_data_dir_shhs("rip_hr", True, True, 2, 1))
    config = SHHSConfig(split = {"train": 1926, "val": 550, "test": 275}, data_type="rip_hr", art_rejection=True, prec_epochs=2, foll_epochs=1)


if __name__ == '__main__':
    main()
