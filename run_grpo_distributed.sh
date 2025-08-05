#!/bin/bash
#
# Launch script for distributed GRPO training
# Launches grpo.py on GPUs 0-6 and actor.py on GPU 7
#

set -e

# Configuration
MODEL_PATH="Qwen/Qwen2.5-1.5B-instruct"
TASK="gsm8k"
STEPS=1000
NUM_ROLLOUTS=7
NUM_PROMPTS=2
GRADIENT_ACCUMULATION_STEPS=2
MAX_GEN_TOKENS=1024
BETA=0.2
CLIP_PARAM=0.2
LR=5e-6
MASTER_ADDR="localhost"
MASTER_PORT="29500"

echo "Starting distributed GRPO training..."
echo "Model: $MODEL_PATH"
echo "Task: $TASK"
echo "Steps: $STEPS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo

# Create logs directory
mkdir -p logs

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$GRPO_PID" ]; then
        kill $GRPO_PID 2>/dev/null || true
    fi
    wait
}
trap cleanup EXIT INT TERM

# Launch unified GRPO script on all GPUs 0-7 (including actor on rank 7)
echo "Launching unified GRPO on GPUs 0-7 (rank 7 acts as actor)..."
deepspeed --include localhost:0,1,2,3,4,5,6,7 grpo.py \
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
    > logs/grpo.log 2>&1 &
GRPO_PID=$!

# No separate actor process needed
ACTOR_PID=""

echo "Both processes launched."
echo "GRPO PID: $GRPO_PID"
echo "Actor PID: $ACTOR_PID"
echo
echo "Monitoring logs (Ctrl-C to stop):"
echo "  GRPO:  tail -f logs/grpo.log"
echo "  Actor: tail -f logs/actor.log"
echo

# Wait for the unified process to complete
wait $GRPO_PID
GRPO_EXIT=$?

echo
echo "Training completed."
echo "GRPO exit code: $GRPO_EXIT"

if [ $GRPO_EXIT -eq 0 ]; then
    echo "Success!"
    exit 0
else
    echo "Training failed!"
    exit 1
fi