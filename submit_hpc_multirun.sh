#!/bin/bash
#BSUB -q hpc
#BSUB -J multirun[1-5]
#BSUB -n 8
#BSUB -W 10:00
#BSUB -u yahei@dtu.dk
#BSUB -R "rusage[mem=2048MB]"
#BSUB -R "span[hosts=1]"
#BSUB -N
#BSUB -o hpc_output/multirun_%J_%I.out
#BSUB -e hpc_output/multirun_%J_%I.err

# MODELS=(logistic_regression lightgbm xgboost mlp)
# MODEL=${MODELS[$((LSB_JOBINDEX - 1))]}
EXPERIMENTS=(
feature_tuning_exp1
feature_tuning_exp2
feature_tuning_exp3
feature_tuning_exp4
feature_tuning_exp5
)
EXPERIMENT=${EXPERIMENTS[$((LSB_JOBINDEX - 1))]}


# DATASETS=(onlyDK1 onlyDK1withoutCrossborder)
# DATASET=${DATASETS[$((LSB_JOBINDEX - 1))]}

mkdir -p hpc_output
cd /zhome/25/9/211757/day_ahead_v2

module load python3/3.12.11
export PATH="$HOME/.local/bin:$PATH"   # so uv is found
uv sync
source .venv/bin/activate

python -m day_ahead_v2.train --config-name=config_test experiments=$EXPERIMENT
