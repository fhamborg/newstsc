#!/bin/bash
###################################
#$ -N copetsa-twitter       # the name of the job
#$ -l h_rt=192:00:00     # max run time
#$ -l h_vmem=15G         # max 4 GB per slot
#$ -o /data/scc/fhamborg/experiments/twitter.out
#$ -e /data/scc/fhamborg/experiments/twitter.err
#$ -m bea
#$ -M felix.hamborg@uni-konstanz.de
#$ -q gpu # to run on nodes with GPGPU
#$ -pe smp 4 # reserve 4 gpu nodes

module load cuda/9.0
module load anaconda
conda activate cope-tsa

echo "starting job"
python controller.py --dataset acl14twitter --experiment_path /data/scc/fhamborg/experiments --continue_run True --num_workers 4
##################################
