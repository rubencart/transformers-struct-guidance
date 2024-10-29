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
#SBATCH -o slrm-%x.%j
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
source /user/leuven/335/vsc33568/.bashrc
conda activate syntra
export PYTHONPATH="$PYTHONPATH":"$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

TRAIN_DATA=/scratch/leuven/335/vsc33568/transformers-struct-guidance/data/tg_train_gen.oracle
DEV_DATA=/scratch/leuven/335/vsc33568/transformers-struct-guidance/data/tg_dev_gen.oracle
MODEL_PATH=/scratch/leuven/335/vsc33568/transformers-struct-guidance/model/tg_$SEED.params

python src/tg_gen.py --train_data "$TRAIN_DATA" --dev_data "$DEV_DATA" --seed "$SEED" --do_train --random_init --model_path "$MODEL_PATH" --architecture "gpt2-large"
#python src/tg_gen.py --train_data "$TRAIN_DATA" --dev_data "$DEV_DATA" --seed "$SEED" --do_train --model_path "$MODEL_PATH" --architecture "gpt2-large" --restore_from "model/tg_last.params" --random_init
