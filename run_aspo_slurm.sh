#!/bin/bash
#
# SLURM sbatch script for simple_GRPO/aspo.py (unified distributed ASPO)
#
# This script requests 8 GPUs on a single node and launches aspo.py on all 8,
# with rank 7 acting as the actor (no separate actor process).
#
# To submit this script:
# sbatch run_aspo_slurm.sh

#SBATCH --job-name=aspo_unified
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # The deepspeed launcher manages processes
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_logs/aspo_unified_%j.out
#SBATCH --error=slurm_logs/aspo_unified_%j.err

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
# Script Configuration (match aspo.py CLI args)
#################################################################
MODEL_PATH="Qwen/Qwen2.5-1.5B-instruct"
TASK="length"
STEPS=1000
NUM_ROLLOUTS=21
NUM_PROMPTS=1
MICRO_BATCH_SIZE=7
GRADIENT_ACCUMULATION_STEPS=3
MAX_GEN_TOKENS=1024
BETA=0.2
CLIP_PARAM=0.2
LR=1e-6
MIN_LEN=16                # ASPO-specific
THETA_ADV=-999999.9       # ASPO-specific
NUM_ROLLOUTS_INITIAL=5    # ASPO-specific
NUM_ROLLOUTS_SPLIT=4      # ASPO-specific
MASTER_ADDR="localhost"
MASTER_PORT="29500"

#################################################################
echo "#################################################################"
echo "Starting DeepSpeed unified ASPO job..."
echo "Model:      $MODEL_PATH"
echo "Task:       $TASK"
echo "Steps:      $STEPS"
echo "Prompts:    $NUM_PROMPTS  |  Rollouts: $NUM_ROLLOUTS"
echo "Micro batch size: $MICRO_BATCH_SIZE"
echo "Grad accum steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Max gen tokens: $MAX_GEN_TOKENS"
echo "Beta:       $BETA"
echo "Clip param: $CLIP_PARAM"
echo "LR:         $LR"
echo "Min len:    $MIN_LEN"
echo "Theta_adv:  $THETA_ADV"
echo "Num rollouts initial: $NUM_ROLLOUTS_INITIAL"
echo "Num rollouts split:   $NUM_ROLLOUTS_SPLIT"
echo "Master:     $MASTER_ADDR:$MASTER_PORT"
echo "#################################################################"

# Launch unified ASPO script on all GPUs 0-7 (including actor on rank 7)
NCCL_DEBUG=INFO deepspeed --include localhost:0,1,2,3,4,5,6,7 aspo.py \
    --model_path "$MODEL_PATH" \
    --task "$TASK" \
    --steps $STEPS \
    --num_rollouts $NUM_ROLLOUTS \
    --num_prompts $NUM_PROMPTS \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --beta $BETA \
    --clip_param $CLIP_PARAM \
    --lr $LR \
    --min_len $MIN_LEN \
    --theta_adv $THETA_ADV \
    --num_rollouts_initial $NUM_ROLLOUTS_INITIAL \
    --num_rollouts_split $NUM_ROLLOUTS_SPLIT \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --job_id "$SLURM_JOB_ID" \
    > logs/aspo_${SLURM_JOB_ID}.log 2>&1

echo "Job finished."