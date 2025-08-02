#!/bin/bash
#
# Sample SLURM sbatch script for simple_GRPO/aspo_vllm_multi_gpu_inference.py
#
# This script requests 8 GPUs on a single node, using 6 for DeepSpeed
# training and 2 for inference actors, as designed in the Python script.
#
# To submit this script:
# sbatch run_aspo_multi_gpu_slurm.sh

#SBATCH --job-name=aspo_multi_gpu
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # The deepspeed launcher manages processes
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_logs/aspo_multi_gpu_%j.out
#SBATCH --error=slurm_logs/aspo_multi_gpu_%j.err

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
export TOTAL_ROLLOUTS=12  # Total rollouts across all inference GPUs
export NUM_PROMPTS=1      # Number of prompts per step
export GRADIENT_ACCUMULATION_STEPS=2
export MAX_GEN_TOKENS=1024
export BETA=0.04
export CLIP_PARAM=0.2
export LR=1e-5

# GPU allocation configuration
# The multi-GPU script uses the last 2 GPUs for inference and the rest for training
NUM_GPUS_REQUESTED=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
NUM_INFERENCE_GPUS=2      # Number of GPUs for inference actors
NUM_TRAIN_GPUS=$((NUM_GPUS_REQUESTED - NUM_INFERENCE_GPUS))

# Create the list of GPU IDs for DeepSpeed training (e.g., "0,1,2,3,4,5")
TRAIN_GPU_IDS=$(seq -s, 0 $((NUM_TRAIN_GPUS - 1)))

# Create the list of GPU IDs for inference actors (e.g., "6,7")
INFERENCE_START_GPU=${NUM_TRAIN_GPUS}
INFERENCE_END_GPU=$((NUM_GPUS_REQUESTED - 1))
ACTOR_GPU_IDS=$(seq -s, ${INFERENCE_START_GPU} ${INFERENCE_END_GPU})

# Path to the script to run
PYTHON_SCRIPT="aspo_vllm_multi_gpu_inference.py"

#################################################################
# Launching the job
#################################################################
echo "#################################################################"
echo "Starting DeepSpeed multi-GPU inference ASPO job..."
echo "Total GPUs on node: ${NUM_GPUS_REQUESTED}"
echo "GPUs for DeepSpeed training: ${NUM_TRAIN_GPUS} (IDs: ${TRAIN_GPU_IDS})"
echo "GPUs for inference actors: ${NUM_INFERENCE_GPUS} (IDs: ${ACTOR_GPU_IDS})"
echo "Total rollouts: ${TOTAL_ROLLOUTS}"
echo "Rollouts per inference GPU: $((TOTAL_ROLLOUTS / NUM_INFERENCE_GPUS))"
echo "Python script: ${PYTHON_SCRIPT}"
echo "Model: ${MODEL_PATH}"
echo "Task: ${TASK}"
echo "Steps: ${STEPS}"
echo "#################################################################"

# Validate configuration
if [ ${NUM_TRAIN_GPUS} -le 0 ]; then
    echo "ERROR: Need at least 1 GPU for training (got ${NUM_TRAIN_GPUS})"
    exit 1
fi

if [ ${NUM_INFERENCE_GPUS} -le 0 ]; then
    echo "ERROR: Need at least 1 GPU for inference (got ${NUM_INFERENCE_GPUS})"
    exit 1
fi

# Check if total rollouts is divisible by number of inference GPUs (for even distribution)
if [ $((TOTAL_ROLLOUTS % NUM_INFERENCE_GPUS)) -ne 0 ]; then
    echo "WARNING: Total rollouts (${TOTAL_ROLLOUTS}) not evenly divisible by inference GPUs (${NUM_INFERENCE_GPUS})"
    echo "Some actors will get extra rollouts"
fi

# The `deepspeed` command will launch `NUM_TRAIN_GPUS` processes.
# The `--include` flag tells deepspeed which GPUs to use on `localhost`.
# The python script (rank 0) will then launch the actor processes on the inference GPUs.
deepspeed --include localhost:${TRAIN_GPU_IDS} \
    ${PYTHON_SCRIPT} \
    --actor_gpus "${ACTOR_GPU_IDS}" \
    --train_gpus ${NUM_TRAIN_GPUS} \
    --total_rollouts ${TOTAL_ROLLOUTS} \
    --num_prompts ${NUM_PROMPTS} \
    --model_path "${MODEL_PATH}" \
    --task "${TASK}" \
    --steps ${STEPS} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --max_gen_tokens ${MAX_GEN_TOKENS} \
    --beta ${BETA} \
    --clip_param ${CLIP_PARAM} \
    --lr ${LR}

echo "Job finished."