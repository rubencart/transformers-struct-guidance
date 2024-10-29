#!/bin/bash
#SBATCH -t 168:00:00
#SBATCH -J lm
#SBATCH -A lp_calculus
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cluster=genius
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_v100_long
#SBATCH -o slrm-%x.%j
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.cartuyvels@kuleuven.be

# Script for running experiments with sbatch
# Example usage:
# sbatch --export="root=/data/leuven/335/vsc33568/transformers-struct-guidance,seed=41,arch=gpt2-large" scripts/run_lm_genius_v100.slurm

PROJECT_ROOT=${root}
SEED=${seed}
ARCH=${arch}

# Define the project directory and set it as the working directory.
cd $PROJECT_ROOT

# Activate Conda environment.
source /user/leuven/335/vsc33568/.bashrc
conda activate syntra
export PYTHONPATH="$PYTHONPATH":"$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

TRAIN_DATA=/scratch/leuven/335/vsc33568/transformers-struct-guidance/data/lm/token/train.txt
DEV_DATA=/scratch/leuven/335/vsc33568/transformers-struct-guidance/data/lm/token/dev.txt
MODEL_PATH=/scratch/leuven/335/vsc33568/transformers-struct-guidance/model/lm_$SEED_$ARCH.params

python src/lm.py --train_data "$TRAIN_DATA" --dev_data "$DEV_DATA" --seed "$SEED" --do_train --model_path "$MODEL_PATH" --architecture "$ARCH" --random_init
#python src/lm.py --train_data "$TRAIN_DATA" --dev_data "$DEV_DATA" --seed "$SEED" --do_train --model_path "$MODEL_PATH" --architecture "$ARCH" --restore_from "model/lm_41_gpt2-large_last.params" --random_init
