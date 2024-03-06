#!/bin/bash
#SBATCH --time=30:0:00
#SBATCH --output=outputs/main4.stdout
#SBATCH --gres=gpu:1

source ~/miniconda/etc/profile.d/conda.sh
conda run -n FYP python -u main4.py
