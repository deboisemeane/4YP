#!/bin/bash
#SBATCH --time=30:0:00
#SBATCH --output=outputs/main2.stdout
#SBATCH --gres=gpu:1

source ~/miniconda/etc/profile.d/conda.sh
conda run -n FYP python main2.py
