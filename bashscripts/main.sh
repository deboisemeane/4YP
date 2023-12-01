#!/bin/bash
#SBATCH --time 1:00
conda activate FYP
python main.py
conda deactivate