#!/bin/bash
#
# Sample SLURM sbatch script for simple_GRPO/grpo_vllm_sync_ds_multi.py
#
# This script requests 8 GPUs on a single node, using 7 for DeepSpeed
# training and 1 for the vLLM actor, as designed in the Python script.
#
# To submit this script:
# sbatch run_grpo_slurm.sh

#SBATCH --job-name=grpo_ds_multi
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # The deepspeed launcher manages processes
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_logs/grpo_ds_multi_%j.out
#SBATCH --error=slurm_logs/grpo_ds_multi_%j.err

# Create directory for slurm logs
mkdir -p slurm_logs

#################################################################
# Environment Setup
#################################################################
echo "Setting up environment..."
conda activate simple_scpo
cd /fsx/home/aginart/dev/rl/simple_GRPO

# For reporting node info
echo "Running on node: $(hostname)"
scontrol show hostnames "$SLURM_JOB_NODELIST"

#################################################################
# Script Configuration
#################################################################
export MODEL_PATH="Qwen/Qwen2.5-1.5B-instruct"
export TASK="length"
export STEPS=1000
export NUM_ROLLOUTS=7
export NUM_PROMPTS=2
export GRADIENT_ACCUMULATION_STEPS=2
export MAX_GEN_TOKENS=1024
export BETA=0.04
export CLIP_PARAM=0.2
export LR=1e-5

# The python script is currently hardcoded for 7 training GPUs via the
# `num_train_devices = 7` variable. This bash script is configured to
# match that (7 train + 1 actor = 8 total GPUs).
# If you change the --gres value, you MUST update the python script.
NUM_GPUS_REQUESTED=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

# We expect one less GPU for training than total requested
NUM_TRAIN_GPUS=$((NUM_GPUS_REQUESTED - 1))

# The last GPU is for the actor
ACTOR_GPU_ID=$((NUM_GPUS_REQUESTED - 1))

# Create the list of GPU IDs for DeepSpeed (e.g., "0,1,2,3,4,5,6")
TRAIN_GPU_IDS=$(seq -s, 0 $((NUM_TRAIN_GPUS - 1)))

# Path to the script to run
PYTHON_SCRIPT="grpo_vllm_sync_ds_multi.py"

#################################################################
# Launching the job
#################################################################
echo "#################################################################"
echo "Starting DeepSpeed job..."
echo "Total GPUs requested: ${NUM_GPUS_REQUESTED}"
echo "GPUs for DeepSpeed training: ${NUM_TRAIN_GPUS} (IDs: ${TRAIN_GPU_IDS})"
echo "GPU for vLLM Actor: ${ACTOR_GPU_ID}"
echo "Python script: ${PYTHON_SCRIPT}"
echo "Model: ${MODEL_PATH}"
echo "Task: ${TASK}"
echo "Steps: ${STEPS}"
echo "#################################################################"



# The `deepspeed` command will launch `NUM_TRAIN_GPUS` processes.
# The `--include` flag tells deepspeed which GPUs to use on `localhost`.
# The python script (rank 0) will then launch the actor process on the `ACTOR_GPU_ID`.
deepspeed --include localhost:${TRAIN_GPU_IDS} \
    ${PYTHON_SCRIPT} \
    --actor_gpu ${ACTOR_GPU_ID} \
    --train_gpus ${NUM_TRAIN_GPUS} \
    --model_path "${MODEL_PATH}" \
    --task "${TASK}" \
    --steps ${STEPS} \
    --num_rollouts ${NUM_ROLLOUTS} \
    --num_prompts ${NUM_PROMPTS} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --max_gen_tokens ${MAX_GEN_TOKENS} \
    --beta ${BETA} \
    --clip_param ${CLIP_PARAM} \
    --lr ${LR}

echo "Job finished."