#!/bin/bash
#BSUB -q hpc
#BSUB -J multirun[1-3]
#BSUB -n 8
#BSUB -W 10:00
#BSUB -u yahei@dtu.dk
#BSUB -R "rusage[mem=2048MB]"
#BSUB -R "span[hosts=1]"
#BSUB -N
#BSUB -o hpc_output/multirun_%J_%I.out
#BSUB -e hpc_output/multirun_%J_%I.err

mkdir -p hpc_output
cd /zhome/25/9/211757/day_ahead_v2

EXPERIMENTS=(alpha_tuning_exp1_hpp
             alpha_tuning_exp2_hpp
             alpha_tuning_exp3_hpp)
EXPERIMENT=${EXPERIMENTS[$LSB_JOBINDEX-1]}

module load python3/3.12.11
export PATH="$HOME/.local/bin:$PATH"   # so uv is found
uv sync
source .venv/bin/activate

python -m day_ahead_v2.train --config-name=config_test experiments=$EXPERIMENT
# python -m day_ahead_v2.wasserstein --config-name=config_test experiments.optimization_parameters.cvar_beta=$CVAR_BETA datasets=$DATASET
