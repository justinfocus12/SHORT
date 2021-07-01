#!/bin/bash
#SBATCH --job-name=duffing
#SBATCH --output=duffing.out
#SBATCH --error=duffing.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=70GB 
#SBATCH --time=03:00:00
#conda activate /scratch/jf4241/penv
python -u duffing_driver.py > duffing_msg.txt
