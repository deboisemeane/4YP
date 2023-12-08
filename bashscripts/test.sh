#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=outputs/test.stdout
#SBATCH --gres=gpu:1

source ~/miniconda/etc/profile.d/conda.sh
conda run -n FYP python test.py
