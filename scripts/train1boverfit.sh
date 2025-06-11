#!/bin/bash
#
#SBATCH --job-name=gan
#SBATCH --output=res_%j.txt
#SBATCH --error=err_%j.txt
#
#SBATCH --partition=submit

#SBATCH --time=8:00:00 
#SBATCH --mem=8G

python ./train1boverfit.py