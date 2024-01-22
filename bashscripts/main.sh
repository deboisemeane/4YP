#!/bin/bash
#SBATCH --time=10:0:00
#SBATCH --output=outputs/main.stdout
##SBATCH --gres=gpu:1

source ~/miniconda/etc/profile.d/conda.sh
conda run -n FYP python main.py
