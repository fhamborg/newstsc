#!/bin/bash

module load cuda
module load anaconda
source activate cope-tsa

echo "starting job"
python controller.py --dataset acl14twitter --experiments_path /data/scc/fhamborg/exp2 --continue_run True --num_workers 7 --results_path results/results_acl14twitter_10
