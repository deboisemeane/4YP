#!/bin/bash
#SBATCH --time=200:0:00
#SBATCH --output=outputs/main3.stdout
#SBATCH --gres=gpu:1

source ~/miniconda/etc/profile.d/conda.sh
conda run -n FYP python -u main3.py
