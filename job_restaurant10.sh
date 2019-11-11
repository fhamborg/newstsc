#!/bin/bash

module load cuda
module load anaconda
source activate ctsacuda10

echo "starting job"
python controller.py --dataset semeval14restaurants --experiments_path /data/scc/fhamborg/exp2 --continue_run True --num_workers 7 --results_path results/results_semeval14restaurants_10

