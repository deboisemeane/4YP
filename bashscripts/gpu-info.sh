#!/bin/bash
#SBATCH --job-name=gpu-check
#SBATCH --time=00:05:00  # Adjust time as needed
#SBATCH --output=gpu-info.txt

nvidia-smi