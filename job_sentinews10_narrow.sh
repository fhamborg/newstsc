#!/bin/bash
###################################
#$ -N copetsa-news       # the name of the job
#$ -l h_rt=72:00:00     # max run time
#$ -l h_vmem=80G         # reserved GB per slot
#$ -o /data/scc/fhamborg/newsenti.out
#$ -e /data/scc/fhamborg/newsenti.err
#$ -m bea
#$ -M felix.hamborg@uni-konstanz.de
#$ -q gpu # to run on nodes with GPGPU
#$ -l gpu=8 # number of reserved GPUs
#$ -pe smp 1 # number of reserved nodes (CPU)

module load cuda
module load anaconda
source activate ctsacuda

echo "starting job"
python controller.py --dataset sentinews --experiments_path /data/scc/fhamborg/exp2 --continue_run True --num_workers -1 --results_path results/results_sentinews_10_narrow
