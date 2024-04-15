#!/bin/bash
#SBATCH --time=100:0:00
#SBATCH --output=outputs/main3.stdout
#SBATCH --gres=gpu:1

source ~/miniconda/etc/profile.d/conda.sh
conda run -n FYP python -u main3.py
