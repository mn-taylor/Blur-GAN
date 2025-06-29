#!/bin/bash
#
#SBATCH --job-name=gan
#SBATCH --output=res_%j.txt
#SBATCH --error=err_%j.txt
#
#SBATCH --partition=submit-gpu
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=8G

nvidia-smi
python ./flow_training_1a_overfit.py