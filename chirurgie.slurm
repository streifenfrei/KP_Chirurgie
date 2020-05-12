#!/bin/bash

#SBATCH --time=08:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=300M   # memory per CPU core
#SBATCH -A p_organdef

OUTFILE="output_chirurgie.txt"
/scratch/ws/0/s3452989-kp_chirurgie/dummy.py  > "$OUTFILE"

exit 0
