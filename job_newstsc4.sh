#!/bin/bash
###################################
#$ -N newstsc4           # the name of the job
#$ -l h_rt=168:00:00     # max run time, see https://www.scc.uni-konstanz.de/summary/current-activities/
#$-l h_vmem=10G          # reserved GB per slot (for each CPU)
#$ -o /data/scc/fhamborg/newstsc4.out
#$ -e /data/scc/fhamborg/newstsc4.err
#$ -m bea
#$ -M felix.hamborg@uni-konstanz.de
#$ -q gpu                # to run on nodes with GPGPU
#$ -l gpu=8              # number of reserved GPUs
#$ -pe smp 8             # number of reserved nodes (8 CPUs), 1 CPU for each GPU (= 8 CPUs in total)

module load cuda
module load anaconda
source activate ctsacuda

echo "starting job"
python controller.py --dataset newstsc --experiments_path /data/scc/fhamborg/exp4 --continue_run True --num_workers -1 --results_path results/results_newstsc4 --cuda_devices SGE_GPU --combi_id 4
