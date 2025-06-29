#!/bin/bash
#
#SBATCH --job-name=flow1
#SBATCH --output=res_%j.txt
#SBATCH --error=err_%j.txt
#
#SBATCH --partition=submit
#SBATCH --time=8:00:00
#SBATCH --mem=8G

nvidia-smi
python ./flow_training1.py