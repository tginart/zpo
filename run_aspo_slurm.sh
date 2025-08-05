#!/bin/bash
#
# SLURM sbatch script for simple_GRPO/aspo.py (unified distributed ASPO)
#
# Requests 8 GPUs on a single node and launches `aspo.py` on all 8 ranks.
# Rank 7 acts as the generation actor; ranks 0-6 are FSDP training workers.
#
# Submit with:
#     sbatch run_aspo_slurm.sh

#SBATCH --job-name=aspo_unified
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # Deepspeed launcher spawns processes
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_logs/aspo_unified_%j.out
#SBATCH --error=slurm_logs/aspo_unified_%j.err

# -----------------------------------------------------------------------------
# Directory / log setup
# -----------------------------------------------------------------------------
mkdir -p slurm_logs
mkdir -p logs

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
echo "Setting up environment..."
conda activate simple_scpo
cd /fsx/home/aginart/dev/rl/simple_GRPO

echo "Running on node: $(hostname)"
scontrol show hostnames "$SLURM_JOB_NODELIST"

# -----------------------------------------------------------------------------
# Script configuration (adjust as needed)
# -----------------------------------------------------------------------------
MODEL_PATH="Qwen/Qwen2.5-1.5B-instruct"
TASK="length"
STEPS=1000
NUM_ROLLOUTS=14          # per prompt
MAX_GEN_TOKENS=1024
BETA=0.2
CLIP_PARAM=0.2
LR=1e-6
MIN_LEN=16
THETA_ADV=-999999.9      # always-split behaviour
MASTER_ADDR="localhost"
MASTER_PORT="29500"

echo "#################################################################"
echo "Starting DeepSpeed unified ASPO job..."
echo "Model:      $MODEL_PATH"
echo "Task:       $TASK"
echo "Steps:      $STEPS"
echo "Prompts:    $NUM_PROMPTS  |  Rollouts: $NUM_ROLLOUTS"
echo "Master:     $MASTER_ADDR:$MASTER_PORT"
echo "#################################################################"

# -----------------------------------------------------------------------------
# Launch – all GPUs 0-7 (actor on rank 7)
# -----------------------------------------------------------------------------
deepspeed --include localhost:0,1,2,3,4,5,6,7 aspo.py \
    --model_path "$MODEL_PATH" \
    --task "$TASK" \
    --steps $STEPS \
    --num_rollouts $NUM_ROLLOUTS \
    --max_gen_tokens $MAX_GEN_TOKENS \
    --beta $BETA \
    --clip_param $CLIP_PARAM \
    --lr $LR \
    --min_len $MIN_LEN \
    --theta_adv $THETA_ADV \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --job_id "$SLURM_JOB_ID" \
    > logs/aspo_${SLURM_JOB_ID}.log 2>&1

echo "Job finished."