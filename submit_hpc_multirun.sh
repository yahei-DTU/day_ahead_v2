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
# EXPERIMENTS=(
# benchmark_exp1_hpp
# benchmark_exp1_windonly
# )
# EXPERIMENT=${EXPERIMENTS[$((LSB_JOBINDEX - 1))]}


# DATASETS=(onlyDK1 onlyDK1withoutCrossborder)
# DATASET=${DATASETS[$((LSB_JOBINDEX - 1))]}

THRESHOLDS=(0.5 0.6 0.7 0.8 0.9)
THRESHOLD=${THRESHOLDS[$((LSB_JOBINDEX - 1))]}

mkdir -p hpc_output
cd /zhome/25/9/211757/day_ahead_v2

module load python3/3.12.11
export PATH="$HOME/.local/bin:$PATH"   # so uv is found
uv sync
source .venv/bin/activate

python -m day_ahead_v2.train --config-name=config_test experiments=alpha_tuning_exp3_hpp "experiments.feature_selection_parameters.shap_cumulative_importance_threshold=${THRESHOLD}"
