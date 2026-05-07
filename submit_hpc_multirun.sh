#!/bin/bash
#BSUB -q hpc
#BSUB -J multirun[1-90]
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
  alpha50_CVaR10_hpp  alpha50_CVaR20_hpp  alpha50_CVaR30_hpp  alpha50_CVaR40_hpp  alpha50_CVaR50_hpp
  alpha50_CVaR60_hpp  alpha50_CVaR70_hpp  alpha50_CVaR80_hpp  alpha50_CVaR90_hpp  alpha50_CVaR100_hpp
  alpha55_CVaR10_hpp  alpha55_CVaR20_hpp  alpha55_CVaR30_hpp  alpha55_CVaR40_hpp  alpha55_CVaR50_hpp
  alpha55_CVaR60_hpp  alpha55_CVaR70_hpp  alpha55_CVaR80_hpp  alpha55_CVaR90_hpp  alpha55_CVaR100_hpp
  alpha60_CVaR10_hpp  alpha60_CVaR20_hpp  alpha60_CVaR30_hpp  alpha60_CVaR40_hpp  alpha60_CVaR50_hpp
  alpha60_CVaR60_hpp  alpha60_CVaR70_hpp  alpha60_CVaR80_hpp  alpha60_CVaR90_hpp  alpha60_CVaR100_hpp
  alpha65_CVaR10_hpp  alpha65_CVaR20_hpp  alpha65_CVaR30_hpp  alpha65_CVaR40_hpp  alpha65_CVaR50_hpp
  alpha65_CVaR60_hpp  alpha65_CVaR70_hpp  alpha65_CVaR80_hpp  alpha65_CVaR90_hpp  alpha65_CVaR100_hpp
  alpha70_CVaR10_hpp  alpha70_CVaR20_hpp  alpha70_CVaR30_hpp  alpha70_CVaR40_hpp  alpha70_CVaR50_hpp
  alpha70_CVaR60_hpp  alpha70_CVaR70_hpp  alpha70_CVaR80_hpp  alpha70_CVaR90_hpp  alpha70_CVaR100_hpp
  alpha75_CVaR10_hpp  alpha75_CVaR20_hpp  alpha75_CVaR30_hpp  alpha75_CVaR40_hpp  alpha75_CVaR50_hpp
  alpha75_CVaR60_hpp  alpha75_CVaR70_hpp  alpha75_CVaR80_hpp  alpha75_CVaR90_hpp  alpha75_CVaR100_hpp
  alpha80_CVaR10_hpp  alpha80_CVaR20_hpp  alpha80_CVaR30_hpp  alpha80_CVaR40_hpp  alpha80_CVaR50_hpp
  alpha80_CVaR60_hpp  alpha80_CVaR70_hpp  alpha80_CVaR80_hpp  alpha80_CVaR90_hpp  alpha80_CVaR100_hpp
  alpha85_CVaR10_hpp  alpha85_CVaR20_hpp  alpha85_CVaR30_hpp  alpha85_CVaR40_hpp  alpha85_CVaR50_hpp
  alpha85_CVaR60_hpp  alpha85_CVaR70_hpp  alpha85_CVaR80_hpp  alpha85_CVaR90_hpp  alpha85_CVaR100_hpp
  alpha90_CVaR10_hpp  alpha90_CVaR20_hpp  alpha90_CVaR30_hpp  alpha90_CVaR40_hpp  alpha90_CVaR50_hpp
  alpha90_CVaR60_hpp  alpha90_CVaR70_hpp  alpha90_CVaR80_hpp  alpha90_CVaR90_hpp  alpha90_CVaR100_hpp
)
EXPERIMENT=${EXPERIMENTS[$((LSB_JOBINDEX - 1))]}

mkdir -p hpc_output
cd /zhome/25/9/211757/day_ahead_v2

module load python3/3.12.11
export PATH="$HOME/.local/bin:$PATH"   # so uv is found
uv sync
source .venv/bin/activate

python -m day_ahead_v2.train --config-name=config_prod "experiments=${EXPERIMENT}"
