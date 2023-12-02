#!/bin/bash
#SBATCH --job-name=gpu-check
#SBATCH --time=00:01:00  # Adjust time as needed
#SBATCH --output=outputs/gpu-info.txt

nvidia-smi