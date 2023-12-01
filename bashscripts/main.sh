#!/bin/bash
#SBATCH --time=1:00
source ~/miniconda/etc/profile.d/conda.sh
conda run -n FYP python main.py
