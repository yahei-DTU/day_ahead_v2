#!/bin/bash
#BSUB -q hpc
#BSUB -J multirun[1-99]
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
alpha_0.5_beta_0.0 alpha_0.5_beta_0.1 alpha_0.5_beta_0.2 alpha_0.5_beta_0.3 alpha_0.5_beta_0.4
alpha_0.5_beta_0.5 alpha_0.5_beta_0.6 alpha_0.5_beta_0.7 alpha_0.5_beta_0.8 alpha_0.5_beta_0.9
alpha_0.5_beta_1.0
alpha_0.55_beta_0.0 alpha_0.55_beta_0.1 alpha_0.55_beta_0.2 alpha_0.55_beta_0.3 alpha_0.55_beta_0.4
alpha_0.55_beta_0.5 alpha_0.55_beta_0.6 alpha_0.55_beta_0.7 alpha_0.55_beta_0.8 alpha_0.55_beta_0.9
alpha_0.55_beta_1.0
alpha_0.6_beta_0.0 alpha_0.6_beta_0.1 alpha_0.6_beta_0.2 alpha_0.6_beta_0.3 alpha_0.6_beta_0.4
alpha_0.6_beta_0.5 alpha_0.6_beta_0.6 alpha_0.6_beta_0.7 alpha_0.6_beta_0.8 alpha_0.6_beta_0.9
alpha_0.6_beta_1.0
alpha_0.65_beta_0.0 alpha_0.65_beta_0.1 alpha_0.65_beta_0.2 alpha_0.65_beta_0.3 alpha_0.65_beta_0.4
alpha_0.65_beta_0.5 alpha_0.65_beta_0.6 alpha_0.65_beta_0.7 alpha_0.65_beta_0.8 alpha_0.65_beta_0.9
alpha_0.65_beta_1.0
alpha_0.7_beta_0.0 alpha_0.7_beta_0.1 alpha_0.7_beta_0.2 alpha_0.7_beta_0.3 alpha_0.7_beta_0.4
alpha_0.7_beta_0.5 alpha_0.7_beta_0.6 alpha_0.7_beta_0.7 alpha_0.7_beta_0.8 alpha_0.7_beta_0.9
alpha_0.7_beta_1.0
alpha_0.75_beta_0.0 alpha_0.75_beta_0.1 alpha_0.75_beta_0.2 alpha_0.75_beta_0.3 alpha_0.75_beta_0.4
alpha_0.75_beta_0.5 alpha_0.75_beta_0.6 alpha_0.75_beta_0.7 alpha_0.75_beta_0.8 alpha_0.75_beta_0.9
alpha_0.75_beta_1.0
alpha_0.8_beta_0.0 alpha_0.8_beta_0.1 alpha_0.8_beta_0.2 alpha_0.8_beta_0.3 alpha_0.8_beta_0.4
alpha_0.8_beta_0.5 alpha_0.8_beta_0.6 alpha_0.8_beta_0.7 alpha_0.8_beta_0.8 alpha_0.8_beta_0.9
alpha_0.8_beta_1.0
alpha_0.85_beta_0.0 alpha_0.85_beta_0.1 alpha_0.85_beta_0.2 alpha_0.85_beta_0.3 alpha_0.85_beta_0.4
alpha_0.85_beta_0.5 alpha_0.85_beta_0.6 alpha_0.85_beta_0.7 alpha_0.85_beta_0.8 alpha_0.85_beta_0.9
alpha_0.85_beta_1.0
alpha_0.9_beta_0.0 alpha_0.9_beta_0.1 alpha_0.9_beta_0.2 alpha_0.9_beta_0.3 alpha_0.9_beta_0.4
alpha_0.9_beta_0.5 alpha_0.9_beta_0.6 alpha_0.9_beta_0.7 alpha_0.9_beta_0.8 alpha_0.9_beta_0.9
alpha_0.9_beta_1.0
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
