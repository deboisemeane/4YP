import torch
from scripts import Train, AdamConfig, SHHSConfig
from src.models import MLP1, Sors
import matplotlib.pyplot as plt
from utils import Timer, get_data_dir_shhs
from debug import AFNet, AFNet_wip


def main():
    print(get_data_dir_shhs("rip_hr", True, True, 2, 1))


if __name__ == '__main__':
    main()
