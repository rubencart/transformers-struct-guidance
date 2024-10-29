#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -J tg
#SBATCH -A lp_calculus
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --ntasks-per-node=18
#SBATCH --cluster=wice
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH -o "slrm.o${SLURM_JOBID}"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.cartuyvels@kuleuven.be

# Script for running experiments with sbatch
# Example usage:
# sbatch --export="root=/data/leuven/335/vsc33568/transformers-struct-guidance,seed=41" scripts/run_tg.slurm

PROJECT_ROOT=${root}
SEED=${seed}

# Define the project directory and set it as the working directory.
cd $PROJECT_ROOT

# Activate Conda environment.
source $HOME/.bashrc
conda activate syntra
export PYTHONPATH="$PYTHONPATH":"$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python -u src/main.py --cfg "$CONFIG_PATH" --seed "$SEED"
python src/tg_gen.py --train_data data/tg_train_gen.oracle --dev_data data/tg_dev_gen.oracle --seed "${SEED}" --do_train --random_init --model_path "model/tg_${SEED}.params" --architecture "gpt2-medium"
