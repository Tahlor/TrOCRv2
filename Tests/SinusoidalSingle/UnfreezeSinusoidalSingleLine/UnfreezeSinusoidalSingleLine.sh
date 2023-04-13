#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu 8000MB
#SBATCH --ntasks 9
#SBATCH --nodes=1
#SBATCH --output="unfreeze-single-sinusoidal.slurm"
#SBATCH --time 03:00:00
#SBATCH --mail-user=jesseclarkwins@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -C 'rhel7&pascal'

#%Module
module purge

export PATH="/home/jclar234/.conda/envs/trocr3/bin:$PATH"
python UnfreezeSinusoidalSingleLine.py 9 23 "/home/jclar234/TrOCR/" "/home/jclar234/TrOCR/Tests/SinusoidalSingle/UnfreezeSinusoidalSingleLine/Results/Model/sub-models"