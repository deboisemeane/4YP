#!/bin/bash
#SBATCH --time=5:0:00
#SBATCH --output=outputs/main.stdout

source ~/miniconda/etc/profile.d/conda.sh
conda run -n FYP python main.py
