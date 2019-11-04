###################################
#!/bin/bash
#$ -N copetsa       # the name of the job
#$ -l h_rt=192:00:00     # 2 days max run time
#$ -l h_vmem=20G         # max 4 GB per slot

module load cuda/9.0
module load anaconda
conda activate cope-tsa

echo "starting job"
python controller.py --dataset acl14twitter --experiment_path /data/scc/fhamborg/experiments --continue_run True
##################################