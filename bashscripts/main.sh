#!/bin/bash
#SBATCH --time=10:00
#SBATCH --output=outputs/main.stdout

source ~/miniconda/etc/profile.d/conda.sh
conda run -n FYP python main.py
