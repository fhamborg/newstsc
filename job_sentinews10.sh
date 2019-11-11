#!/bin/bash

module load cuda
module load anaconda
source activate ctsacuda

echo "starting job"
python controller.py --dataset sentinews --experiments_path /data/scc/fhamborg/exp2 --continue_run True --num_workers 7 --results_path results/results_sentinews_10

