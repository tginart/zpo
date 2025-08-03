#!/bin/bash
#
# Sample SLURM sbatch script for simple_GRPO/aspo_vllm_async_simplified.py
#
# This script requests 8 GPUs on a single node, using 7 for DeepSpeed
# training and 1 for the vLLM actor, as designed in the Python script.
#
# To submit this script:
# sbatch run_grpo_slurm.sh

#SBATCH --job-name=aspo_async_simple
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # The deepspeed launcher manages processes
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_logs/aspo_ds_multi_%j.out
#SBATCH --error=slurm_logs/aspo_ds_multi_%j.err

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
# The simplified script always uses 1 prompt and 28 rollouts; NUM_PROMPTS arg is not needed
export GRADIENT_ACCUMULATION_STEPS=3
export MAX_GEN_TOKENS=1024
export BETA=0.04
export CLIP_PARAM=0.2
export LR=1e-5
# export K=
# export THETA_A=
# export MIN_LEN=

# The python script is currently hardcoded for 7 training GPUs via the
# `num_train_devices = 7` variable. This bash script is configured to
# match that (7 train + 1 actor = 8 total GPUs).
# We pass all 8 GPUs to DeepSpeed but only use 7 for training.
# The actor process will use GPU 7 directly.
NUM_GPUS_REQUESTED=8
NUM_TRAIN_GPUS=7
ACTOR_GPU_ID=7

# Path to the script to run
PYTHON_SCRIPT="aspo_vllm_async_simplified.py"

#################################################################
# Launching the job
#################################################################
echo "#################################################################"
echo "Starting DeepSpeed async-simple ASPO job..."
echo "Total GPUs on node: ${NUM_GPUS_REQUESTED}"
echo "GPUs for DeepSpeed training: ${NUM_TRAIN_GPUS} (IDs: 0,1,2,3,4,5,6)"
echo "GPU for vLLM Actor: ${ACTOR_GPU_ID}"
echo "Python script: ${PYTHON_SCRIPT}"
echo "Model: ${MODEL_PATH}"
echo "Task: ${TASK}"
echo "Steps: ${STEPS}"
echo "#################################################################"



# The `deepspeed` command will launch `NUM_TRAIN_GPUS` processes.
# We pass all 8 GPUs to DeepSpeed but only use 7 for training.
# The python script (rank 0) will then launch the actor process on GPU 7.
deepspeed --num_gpus 7 \
    ${PYTHON_SCRIPT} \
    --actor_gpu ${ACTOR_GPU_ID} \
    --train_gpus ${NUM_TRAIN_GPUS} \
    --model_path "${MODEL_PATH}" \
    --task "${TASK}" \
    --steps ${STEPS} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --max_gen_tokens ${MAX_GEN_TOKENS} \
    --beta ${BETA} \
    --clip_param ${CLIP_PARAM} \
    --lr ${LR}

echo "Job finished."