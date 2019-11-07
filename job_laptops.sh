#!/bin/bash
###################################
#$ -N copetsa-laptops       # the name of the job
#$ -l h_rt=192:00:00     # max run time
#$ -l h_vmem=15G         # max 4 GB per slot
#$ -o /data/scc/fhamborg/experiments/laptops.out
#$ -e /data/scc/fhamborg/experiments/laptops.err
#$ -m bea
#$ -M felix.hamborg@uni-konstanz.de
#$ -q gpu # to run on nodes with GPGPU
#$ -pe smp 2 # reserve 4 gpu nodes

module load cuda/9.0
module load anaconda
conda activate ctsa

echo "starting job"
python controller.py --dataset semeval14laptops --experiments_path /data/scc/fhamborg/experiments --continue_run True --num_workers 2
##################################
