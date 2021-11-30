#!/bin/bash
#SBATCH --job-name=hm
#SBATCH --output=hm.out
#SBATCH --error=hm.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=70GB 
#SBATCH --time=05:00:00
#conda activate /scratch/jf4241/penv
python -u hm_bothcast_driver.py > hm_msg.txt
