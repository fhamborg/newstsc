#!/bin/bash
###################################
#$ -N newstsc0           # the name of the job
#$ -l h_rt=168:00:00     # max run time, see https://www.scc.uni-konstanz.de/summary/current-activities/
#$-l h_vmem=10G          # reserved GB per slot (for each CPU)
#$ -o /data/scc/fhamborg/newstsc0.out
#$ -e /data/scc/fhamborg/newstsc0.err
#$ -m bea
#$ -M felix.hamborg@uni-konstanz.de
#$ -q gpu                # to run on nodes with GPGPU
#$ -l gpu=7              # number of reserved GPUs
#$ -pe smp 7             # number of reserved nodes (8 CPUs), 1 CPU for each GPU (= 8 CPUs in total)

module load cuda
module load anaconda
source activate ctsacuda

echo "starting job"
python controller.py --dataset newstsc2 --experiments_path /data/scc/fhamborg/exp0 --continue_run True --num_workers -1 --results_path results/results_newstsc0 --cuda_devices SGE_GPU --combi_mode default --combi_id 0
echo "finished job"
