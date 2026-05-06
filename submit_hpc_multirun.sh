#!/bin/bash
#BSUB -q hpc
#BSUB -J multirun[1-10]
#BSUB -n 20
#BSUB -W 20:00
#BSUB -u yahei@dtu.dk
#BSUB -R "rusage[mem=2048MB]"
#BSUB -R "span[hosts=1]"
#BSUB -N
#BSUB -o hpc_output/multirun_%J_%I.out
#BSUB -e hpc_output/multirun_%J_%I.err

# MODELS=(logistic_regression lightgbm xgboost mlp)
# MODEL=${MODELS[$((LSB_JOBINDEX - 1))]}
EXPERIMENTS=(alpha50_CVaR50_hpp alpha60_CVaR50_hpp alpha70_CVaR50_hpp alpha80_CVaR50_hpp alpha90_CVaR50_hpp alpha50_CVaR100_hpp alpha60_CVaR100_hpp alpha70_CVaR100_hpp alpha80_CVaR100_hpp alpha90_CVaR100_hpp)
EXPERIMENT=${EXPERIMENTS[$((LSB_JOBINDEX - 1))]}

mkdir -p hpc_output
cd /zhome/25/9/211757/day_ahead_v2

module load python3/3.12.11
export PATH="$HOME/.local/bin:$PATH"   # so uv is found
uv sync
source .venv/bin/activate

python -m day_ahead_v2.train --config-name=config_prod "experiments=${EXPERIMENT}"
