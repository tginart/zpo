#!/bin/bash
#
# SLURM sbatch script for simple_GRPO/gspo.py (unified distributed GSPO)
#
# This script requests 8 GPUs on a single node and launches gspo.py on all 8,
# with rank 7 acting as the actor (no separate actor process).
#
# To submit this script:
# sbatch run_gspo_slurm.sh

#SBATCH --job-name=gspo_unified
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # The deepspeed launcher manages processes
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_logs/gspo_unified_%j.out
#SBATCH --error=slurm_logs/gspo_unified_%j.err

# Create directory for slurm logs and training logs
mkdir -p slurm_logs
mkdir -p logs

#################################################################
# Environment Setup
#################################################################
echo "Setting up environment..."
conda activate simple_scpo
cd /fsx/home/aginart/dev/rl/simple_GRPO

echo "Running on node: $(hostname)"
scontrol show hostnames "$SLURM_JOB_NODELIST"

#################################################################
# Script Configuration (match run_grpo_distributed.sh)
#################################################################
MODEL_PATH="Qwen/Qwen2.5-1.5B-instruct"
TASK="gsm8k"
STEPS=2000
NUM_ROLLOUTS=14
NUM_PROMPTS=1
GRADIENT_ACCUMULATION_STEPS=2
MAX_GEN_TOKENS=1024
BETA=0.01
CLIP_PARAM=0.1
LR=1e-5
MASTER_ADDR="localhost"
MASTER_PORT="29500"

echo "#################################################################"
echo "Starting DeepSpeed unified GSPO job..."
echo "Model: $MODEL_PATH"
echo "Task: $TASK"
echo "Steps: $STEPS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "#################################################################"

# Launch unified GSPO script on all GPUs 0-7 (including actor on rank 7)
deepspeed --include localhost:0,1,2,3,4,5,6,7 gspo.py \
    --model_path "$MODEL_PATH" \
    --task "$TASK" \
    --steps $STEPS \
    --num_rollouts $NUM_ROLLOUTS \
    --num_prompts $NUM_PROMPTS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --beta $BETA \
    --clip_param $CLIP_PARAM \
    --lr $LR \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --job_id "$SLURM_JOB_ID" \
    > logs/gspo_${TASK}_${SLURM_JOB_ID}.log 2>&1

echo "Job finished." 