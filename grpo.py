"""
GRPO trainer script - runs on GPUs 0-6 with FSDP
Communicates with actor.py via torch.distributed
"""

import os, random, time, json, argparse
from typing import List, Dict, Any
import pickle

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from tqdm import trange

from tasks import get_task

# ----------------------------------------------------------------------------------
# CLI Arguments
# ----------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="(internal) local rank passed by launcher")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-instruct")
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--num_rollouts", type=int, default=7)
    parser.add_argument("--num_prompts", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_gen_tokens", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--clip_param", type=float, default=0.1)    
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    
    # Coordination parameters
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="29500")
    
    return parser.parse_args()

args = get_args()

# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------

task = args.task
model_path = args.model_path
num_rollouts = args.num_rollouts
num_prompts = args.num_prompts
gradient_accumulation_steps = args.gradient_accumulation_steps
max_gen_tokens = args.max_gen_tokens
beta = args.beta
clip_param = args.clip_param
lr = args.lr
grad_clip = args.grad_clip
num_train_devices = 7

# Batch size calculations
total_batch_size = num_prompts * num_rollouts
if total_batch_size % gradient_accumulation_steps != 0:
    raise ValueError("num_prompts * num_rollouts must be divisible by gradient_accumulation_steps")

# FSDP mixed precision config
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# ----------------------------------------------------------------------------------
# Communication utilities
# ----------------------------------------------------------------------------------

def send_object(obj, dst_rank, tag=0):
    """Send a small control message/metadata to another rank via torch.distributed.
    
    WARNING: Only use for small control messages (acks, requests, metadata).
    For heavy payloads like state dicts or batches, use tensor-based communication!
    """
    data = pickle.dumps(obj)
    
    # Send size first
    size_tensor = torch.tensor([len(data)], dtype=torch.long, device=torch.cuda.current_device())
    dist.send(size_tensor, dst=dst_rank, tag=tag)
    
    # Send data
    data_tensor = torch.tensor(list(data), dtype=torch.uint8, device=torch.cuda.current_device())
    dist.send(data_tensor, dst=dst_rank, tag=tag)

def recv_object(src_rank, tag=0):
    """Receive a small control message/metadata from another rank via torch.distributed.
    
    WARNING: Only use for small control messages (acks, requests, metadata).
    For heavy payloads like state dicts or batches, use tensor-based communication!
    """
    device = torch.cuda.current_device()
    
    # Receive size
    size_tensor = torch.empty([1], dtype=torch.long, device=device)
    dist.recv(size_tensor, src=src_rank, tag=tag)
    size = size_tensor[0].item()
    
    # Receive data
    data_tensor = torch.empty([size], dtype=torch.uint8, device=device)
    dist.recv(data_tensor, src=src_rank, tag=tag)
    
    # Deserialize
    data = bytes(data_tensor.cpu().tolist())
    return pickle.loads(data)

def send_state_dict(state_dict, dst_rank, tag=0):
    """Send a state dict using direct tensor communication (much faster than pickle)"""
    device = torch.cuda.current_device()
    
    # Send number of tensors
    num_tensors = len(state_dict)
    count_tensor = torch.tensor([num_tensors], dtype=torch.long, device=device)
    dist.send(count_tensor, dst=dst_rank, tag=tag)
    
    # Send each tensor
    tag_offset = 1
    for i, (name, tensor) in enumerate(state_dict.items()):
        # Send tensor name length and name
        name_bytes = name.encode('utf-8')
        name_len = torch.tensor([len(name_bytes)], dtype=torch.long, device=device)
        dist.send(name_len, dst=dst_rank, tag=tag + tag_offset)
        tag_offset += 1
        
        name_tensor = torch.tensor(list(name_bytes), dtype=torch.uint8, device=device)
        dist.send(name_tensor, dst=dst_rank, tag=tag + tag_offset)
        tag_offset += 1
        
        # Send tensor shape length and shape
        shape = list(tensor.shape)
        shape_len = torch.tensor([len(shape)], dtype=torch.long, device=device)
        dist.send(shape_len, dst=dst_rank, tag=tag + tag_offset)
        tag_offset += 1
        
        shape_tensor = torch.tensor(shape, dtype=torch.long, device=device)
        dist.send(shape_tensor, dst=dst_rank, tag=tag + tag_offset)
        tag_offset += 1
        
        # Send tensor dtype info
        dtype_info = torch.tensor([0 if tensor.dtype == torch.bfloat16 else 1], dtype=torch.long, device=device)
        dist.send(dtype_info, dst=dst_rank, tag=tag + tag_offset)
        tag_offset += 1
        
        # Send tensor data directly (keep on GPU if possible)
        tensor_gpu = tensor.to(device)
        dist.send(tensor_gpu.contiguous(), dst=dst_rank, tag=tag + tag_offset)
        tag_offset += 1

def recv_state_dict(src_rank, tag=0):
    """Receive a state dict using direct tensor communication"""
    device = torch.cuda.current_device()
    
    # Receive number of tensors
    count_tensor = torch.empty([1], dtype=torch.long, device=device)
    dist.recv(count_tensor, src=src_rank, tag=tag)
    num_tensors = count_tensor[0].item()
    
    state_dict = {}
    
    # Receive each tensor
    tag_offset = 1
    for i in range(num_tensors):
        # Receive tensor name
        name_len_tensor = torch.empty([1], dtype=torch.long, device=device)
        dist.recv(name_len_tensor, src=src_rank, tag=tag + tag_offset)
        tag_offset += 1
        name_len = name_len_tensor[0].item()
        
        name_tensor = torch.empty([name_len], dtype=torch.uint8, device=device)
        dist.recv(name_tensor, src=src_rank, tag=tag + tag_offset)
        tag_offset += 1
        name = bytes(name_tensor.cpu().tolist()).decode('utf-8')
        
        # Receive tensor shape
        shape_len_tensor = torch.empty([1], dtype=torch.long, device=device)
        dist.recv(shape_len_tensor, src=src_rank, tag=tag + tag_offset)
        tag_offset += 1
        shape_len = shape_len_tensor[0].item()
        
        shape_tensor = torch.empty([shape_len], dtype=torch.long, device=device)
        dist.recv(shape_tensor, src=src_rank, tag=tag + tag_offset)
        tag_offset += 1
        actual_shape = [dim.item() for dim in shape_tensor]
        
        # Receive tensor dtype info
        dtype_tensor = torch.empty([1], dtype=torch.long, device=device)
        dist.recv(dtype_tensor, src=src_rank, tag=tag + tag_offset)
        tag_offset += 1
        dtype = torch.bfloat16 if dtype_tensor[0].item() == 0 else torch.float32
        
        # Receive tensor data
        tensor = torch.empty(actual_shape, dtype=dtype, device=device)
        dist.recv(tensor, src=src_rank, tag=tag + tag_offset)
        tag_offset += 1
        
        state_dict[name] = tensor
    
    return state_dict

def send_batch(batch, dst_rank, tag=0):
    """Send a batch using direct tensor communication (much faster than pickle)"""
    device = torch.cuda.current_device()
    
    # Convert lists to tensors and send them
    sequences = torch.tensor(batch["sequences"], dtype=torch.long, device=device)
    rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
    gen_logps = torch.tensor(batch["gen_logps"], dtype=torch.float32, device=device)
    ref_logps = torch.tensor(batch["ref_logps"], dtype=torch.float32, device=device)
    
    # Send tensor shapes first
    dist.send(torch.tensor(sequences.shape, dtype=torch.long, device=device), dst=dst_rank, tag=tag)
    dist.send(torch.tensor(rewards.shape, dtype=torch.long, device=device), dst=dst_rank, tag=tag+1)
    dist.send(torch.tensor(gen_logps.shape, dtype=torch.long, device=device), dst=dst_rank, tag=tag+2)
    dist.send(torch.tensor(ref_logps.shape, dtype=torch.long, device=device), dst=dst_rank, tag=tag+3)
    
    # Send tensors
    dist.send(sequences, dst=dst_rank, tag=tag+10)
    dist.send(rewards, dst=dst_rank, tag=tag+11)
    dist.send(gen_logps, dst=dst_rank, tag=tag+12)
    dist.send(ref_logps, dst=dst_rank, tag=tag+13)
    
    # Send metadata using send_object (small control data)
    metadata = {
        "prompt_lens": batch["prompt_lens"],
        "prompts": batch["prompts"],
        "answers": batch["answers"],
        "ans_token_ids": batch["ans_token_ids"],
        "gen_time": batch["gen_time"]
    }
    send_object(metadata, dst_rank, tag=tag+20)

def recv_batch(src_rank, tag=0):
    """Receive a batch using direct tensor communication"""
    device = torch.cuda.current_device()
    
    # Receive tensor shapes
    seq_shape_tensor = torch.empty([2], dtype=torch.long, device=device)
    reward_shape_tensor = torch.empty([1], dtype=torch.long, device=device)
    gen_logps_shape_tensor = torch.empty([2], dtype=torch.long, device=device)
    ref_logps_shape_tensor = torch.empty([2], dtype=torch.long, device=device)
    
    dist.recv(seq_shape_tensor, src=src_rank, tag=tag)
    dist.recv(reward_shape_tensor, src=src_rank, tag=tag+1)
    dist.recv(gen_logps_shape_tensor, src=src_rank, tag=tag+2)
    dist.recv(ref_logps_shape_tensor, src=src_rank, tag=tag+3)
    
    # Receive tensors
    sequences = torch.empty(seq_shape_tensor.tolist(), dtype=torch.long, device=device)
    rewards = torch.empty(reward_shape_tensor.tolist(), dtype=torch.float32, device=device)
    gen_logps = torch.empty(gen_logps_shape_tensor.tolist(), dtype=torch.float32, device=device)
    ref_logps = torch.empty(ref_logps_shape_tensor.tolist(), dtype=torch.float32, device=device)
    
    dist.recv(sequences, src=src_rank, tag=tag+10)
    dist.recv(rewards, src=src_rank, tag=tag+11)
    dist.recv(gen_logps, src=src_rank, tag=tag+12)
    dist.recv(ref_logps, src=src_rank, tag=tag+13)
    
    # Receive metadata
    metadata = recv_object(src_rank, tag=tag+20)
    
    # Reconstruct batch
    batch = {
        "sequences": sequences.cpu().tolist(),
        "rewards": rewards.cpu().tolist(),
        "gen_logps": gen_logps.cpu().tolist(),
        "ref_logps": ref_logps.cpu().tolist(),
        "prompt_lens": metadata["prompt_lens"],
        "prompts": metadata["prompts"],
        "answers": metadata["answers"],
        "ans_token_ids": metadata["ans_token_ids"],
        "gen_time": metadata["gen_time"]
    }
    
    return batch

# ----------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------

def per_token_logps(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return torch.gather(logp, -1, ids.unsqueeze(-1)).squeeze(-1)

# ----------------------------------------------------------------------------------
# Actor role function
# ----------------------------------------------------------------------------------

def run_actor_role(actor_rank, world_size, model_path, task_name, master_addr, master_port):
    """Run the actor role for rank 7"""
    print(f"[ACTOR] Starting actor role on rank {actor_rank}")
    
    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"[ACTOR] Loading models...")
    device = torch.cuda.current_device()
    model_gen = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model_ref = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model_gen.eval()
    model_ref.eval()
    print(f"[ACTOR] Models loaded")
    
    # Load task
    task_items, reward_fn = get_task(task_name)
    print(f"[ACTOR] Task '{task_name}' loaded with {len(task_items)} items")
    
    # System prompt
    system_prompt = (
        "You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags."
    )
    
    # Signal readiness to training ranks
    print(f"[ACTOR] Signaling readiness (barrier)...")
    dist.barrier()
    print(f"[ACTOR] Ready for training loop")
    
    step = 0
    try:
        while True:
            step += 1
            print(f"[ACTOR] Step {step}: Waiting for shards from all trainer ranks...")
            
            # Receive state dict from rank 0 using proper tensor communication
            print(f"[ACTOR] Step {step}: Waiting for control message from rank 0...")
            msg = recv_object(src_rank=0, tag=step*10)
            print(f"[ACTOR] Step {step}: Received control message from rank 0")
            
            if msg["type"] == "STOP":
                print(f"[ACTOR] Received STOP signal, shutting down")
                return
            elif msg["type"] == "UPDATE_WEIGHTS":
                print(f"[ACTOR] Step {step}: Receiving {msg['num_tensors']} tensors...")
                
                # Receive each tensor using proper PyTorch distributed communication
                state_dict = {}
                for i in range(msg["num_tensors"]):
                    # Receive tensor metadata
                    tensor_info = recv_object(src_rank=0, tag=step*10 + i*3 + 1)
                    name = tensor_info["name"]
                    shape = tensor_info["shape"]
                    dtype_str = tensor_info["dtype"]
                    
                    # Convert dtype string back to dtype
                    if "bfloat16" in dtype_str:
                        dtype = torch.bfloat16
                    elif "float32" in dtype_str:
                        dtype = torch.float32
                    elif "int64" in dtype_str:
                        dtype = torch.int64
                    else:
                        dtype = torch.float32  # fallback
                    
                    # Receive actual tensor using dist.recv
                    tensor = torch.empty(shape, dtype=dtype, device=device)
                    dist.recv(tensor, src=0, tag=step*10 + i*3 + 2)
                    
                    state_dict[name] = tensor
                
                print(f"[ACTOR] Step {step}: Received all tensors, loading into model...")
                model_gen.load_state_dict(state_dict)
                print(f"[ACTOR] Step {step}: Weights updated")
            else:
                raise RuntimeError(f"Unknown message type: {msg['type']}")
            
            # Send acknowledgment to rank 0
            send_object({"status": "OK"}, dst_rank=0, tag=step*10+100)
            print(f"[ACTOR] Step {step}: Sent ack to rank 0")
            
            # Wait for generation request
            print(f"[ACTOR] Step {step}: Waiting for generation request...")
            gen_msg = recv_object(src_rank=0, tag=step*10+2)
            print(f"[ACTOR] Step {step}: Received generation request: {gen_msg}")
            assert gen_msg["type"] == "GENERATE", f"Expected GENERATE, got {gen_msg['type']}"
            
            # Extract parameters
            num_rollouts = gen_msg["num_rollouts"]
            num_prompts = gen_msg["num_prompts"]
            max_gen_tokens = gen_msg["max_gen_tokens"]
            
            print(f"[ACTOR] Step {step}: Generating {num_rollouts} rollouts for {num_prompts} prompts...")
            tic = time.time()
            
            # Generate batch
            batch = generate_batch(
                model_gen, model_ref, tokenizer, reward_fn, task_items,
                num_prompts, num_rollouts, max_gen_tokens, system_prompt, device
            )
            
            gen_time = time.time() - tic
            batch["gen_time"] = gen_time
            
            print(f"[ACTOR] Step {step}: Generated batch in {gen_time:.2f}s, sending to rank 0...")
            
            # Send batch back to rank 0 using tensor communication
            print(f"[ACTOR] Step {step}: Sending batch tensors to rank 0...")
            
            # Convert batch data to tensors and send using dist.send
            sequences = torch.tensor(batch["sequences"], dtype=torch.long, device=device)
            rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
            gen_logps = torch.tensor(batch["gen_logps"], dtype=torch.float32, device=device)
            ref_logps = torch.tensor(batch["ref_logps"], dtype=torch.float32, device=device)
            
            # Send tensor shapes first as control message
            send_object({
                "shapes": {
                    "sequences": list(sequences.shape),
                    "rewards": list(rewards.shape),
                    "gen_logps": list(gen_logps.shape),
                    "ref_logps": list(ref_logps.shape)
                },
                "metadata": {
                    "prompt_lens": batch["prompt_lens"],
                    "prompts": batch["prompts"],
                    "answers": batch["answers"],
                    "ans_token_ids": batch["ans_token_ids"],
                    "gen_time": batch["gen_time"]
                }
            }, dst_rank=0, tag=step*10+3)
            
            # Send tensors
            dist.send(sequences, dst=0, tag=step*10+10)
            dist.send(rewards, dst=0, tag=step*10+11) 
            dist.send(gen_logps, dst=0, tag=step*10+12)
            dist.send(ref_logps, dst=0, tag=step*10+13)
            
            print(f"[ACTOR] Step {step}: Batch sent")
    except Exception as e:
        print(f"[ACTOR] Error in training loop: {e}")
        raise
    
    print(f"[ACTOR] Actor shutting down")

def generate_batch(model_gen, model_ref, tokenizer, reward_fn, task_items, 
                  num_prompts, num_rollouts, max_gen_tokens, system_prompt, device):
    """Generate a batch of rollouts - copied from actor.py"""
    
    # Sample prompts
    items = random.sample(task_items, num_prompts)
    
    all_sequences = []
    all_rewards = []
    all_gen_logps = []
    all_ref_logps = []
    all_answers = []
    all_ans_token_ids = []
    all_prompts = []
    all_prompt_lens = []
    
    for item in items:
        user_prompt = item["prompt"][0]["content"] if isinstance(item["prompt"], list) else item["prompt"]
        all_prompts.append(user_prompt)
        
        # Create chat template
        tip_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = tokenizer(tip_text, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Extend prompt_lens for each rollout
        all_prompt_lens.extend([prompt_len] * num_rollouts)
        
        # Generate sequences
        with torch.no_grad():
            full_sequences_ids = model_gen.generate(
                **inputs,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                max_new_tokens=max_gen_tokens,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=num_rollouts,
            )
        
        # Extract answers (completion only)
        ans_token_ids = full_sequences_ids[:, prompt_len:]
        answers = tokenizer.batch_decode(ans_token_ids, skip_special_tokens=True)
        
        # Get rewards
        rewards = reward_fn(answers, items=[item] * len(answers))
        
        # Get log probabilities
        def get_logprobs_for_generated(model, full_ids, p_len):
            with torch.no_grad():
                logits = model(full_ids).logits
            log_softmax = torch.log_softmax(logits, dim=-1)
            log_probs = torch.gather(log_softmax[:, :-1, :], -1, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            return log_probs[:, p_len - 1 :]
        
        gen_logps = get_logprobs_for_generated(model_gen, full_sequences_ids, prompt_len)
        ref_logps = get_logprobs_for_generated(model_ref, full_sequences_ids, prompt_len)
        
        # Accumulate results
        all_sequences.extend(full_sequences_ids.tolist())
        all_rewards.extend(rewards)
        all_gen_logps.extend(gen_logps.cpu().tolist())
        all_ref_logps.extend(ref_logps.cpu().tolist())
        all_answers.extend(answers)
        all_ans_token_ids.extend(ans_token_ids.cpu().tolist())
    
    # Pad sequences to same length
    max_seq_len = max(len(s) for s in all_sequences)
    for seq in all_sequences:
        seq.extend([tokenizer.pad_token_id] * (max_seq_len - len(seq)))
    
    # Pad logp tensors to common generation length
    max_gen_len = max(len(l) for l in all_gen_logps)
    for l in all_gen_logps:
        l.extend([0.0] * (max_gen_len - len(l)))
    for l in all_ref_logps:
        l.extend([0.0] * (max_gen_len - len(l)))
    
    return {
        "sequences": all_sequences,
        "rewards": all_rewards,
        "gen_logps": all_gen_logps,
        "ref_logps": all_ref_logps,
        "prompt_lens": all_prompt_lens,
        "prompts": all_prompts,
        "answers": all_answers,
        "ans_token_ids": all_ans_token_ids,
    }

# ----------------------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------------------

def main():
    # Initialize distributed training
    # We expect to be launched with deepspeed --include localhost:0,1,2,3,4,5,6
    # which will give us ranks 0-6 in a world of size 7
    # We need to extend this to communicate with rank 7 (the actor)
    
    # Set up environment for distributed training
    print(f"[GRPO] Setting up environment for distributed training")
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    print(f"[GRPO] Environment set: MASTER_ADDR={os.environ['MASTER_ADDR']} MASTER_PORT={os.environ['MASTER_PORT']}")
    
    # Initialize distributed training directly with extended world (ranks 0-7, where 7 is the actor)
    print(f"[GRPO] Initializing distributed training for extended world (ranks 0-7)")
    
    # Get local rank from environment (set by deepspeed launcher)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = 8  # 7 trainers + 1 actor
    
    torch.cuda.set_device(local_rank)
    
    print(f"[GRPO] Initializing process group: rank={global_rank}, world_size={world_size}")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        world_size=world_size,
        rank=global_rank
    )
    print(f"[GRPO] Process group initialized: rank {global_rank}/{world_size}")
    
    # Use global_rank as extended_rank for compatibility
    extended_rank = global_rank
    extended_world_size = world_size
    
    # Rank 7 acts as the actor, others as trainers
    if extended_rank == 7:
        print(f"[GRPO] Rank {extended_rank} will act as ACTOR")
        run_actor_role(extended_rank, extended_world_size, model_path, task, args.master_addr, args.master_port)
        return
    else:
        print(f"[GRPO] Rank {extended_rank} will act as TRAINER")
        
    # Create a separate process group for training ranks only (0-6)
    trainer_ranks = list(range(7))  # [0, 1, 2, 3, 4, 5, 6]
    trainer_group = dist.new_group(trainer_ranks)
    print(f"[GRPO] Created trainer process group with ranks {trainer_ranks}")
    
    print(f"[GRPO] Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model on CPU first, then move to FSDP
    print(f"[GRPO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    
    # Determine the transformer layer class for wrapping policy
    # For Qwen2.5, this should be Qwen2DecoderLayer
    try:
        from transformers.models.qwen2 import Qwen2DecoderLayer
        transformer_layer_cls = Qwen2DecoderLayer
    except ImportError:
        # Fallback: wrap every module 
        transformer_layer_cls = torch.nn.Module
        print(f"[GRPO] Warning: Could not import Qwen2DecoderLayer, using generic wrapping")
    
    # Wrap model with FSDP (using trainer group only)
    print(f"[GRPO] Initializing FSDP...")
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=ModuleWrapPolicy({transformer_layer_cls}),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        process_group=trainer_group,  # Use only ranks 0-6
    )
    
    # Create optimizer
    print(f"[GRPO] Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    print(f'[GRPO] Rank {extended_rank} FSDP model and optimizer created.')

    if extended_rank == 0:
        print(f"[GRPO] Creating SummaryWriter and logging args")
        writer = SummaryWriter(f"runs/grpo_distributed_{time.strftime('%Y%m%d-%H%M%S')}")
        # Log args
        writer.add_text("args", json.dumps(vars(args)))

    # Wait for actor to be ready
    print(f"[GRPO] Rank {extended_rank} waiting for actor to be ready (barrier)...")
    dist.barrier()  # Actor will join this barrier when ready
    print(f"[GRPO] Rank {extended_rank} actor is ready, starting training loop")

    # Training loop
    for step in trange(1, args.steps + 1, desc="Training Steps", disable=(extended_rank != 0)):
        
        # Only rank 0 gets the full state dict and sends tensors to actor
        if extended_rank == 0:
            print(f"[GRPO] Rank {extended_rank}: Getting full state_dict for step {step}...")
            # Use full state dict on rank 0 - FSDP will gather from all ranks
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                full_state_dict = model.state_dict()
            print(f"[GRPO] Rank {extended_rank}: Got full state_dict for step {step}")
            
            print(f"[GRPO] Rank {extended_rank}: Sending state dict to actor...")
            
            # Send control message first
            send_object({"type": "UPDATE_WEIGHTS", "num_tensors": len(full_state_dict)}, 
                       dst_rank=7, tag=step*10)
            
            # Send each tensor using proper PyTorch distributed communication
            for i, (name, tensor) in enumerate(full_state_dict.items()):
                # Send tensor name as small control message
                send_object({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}, 
                           dst_rank=7, tag=step*10 + i*3 + 1)
                
                # Send actual tensor data using dist.send
                tensor_gpu = tensor.cuda()  # Ensure on GPU for NCCL
                dist.send(tensor_gpu, dst=7, tag=step*10 + i*3 + 2)
            
            print(f"[GRPO] Rank {extended_rank}: Sent state dict to actor")
        else:
            # Other ranks just participate in the collective gather
            print(f"[GRPO] Rank {extended_rank}: Participating in state_dict gather for step {step}")
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                model.state_dict()  # Participate in collective but don't use result
            print(f"[GRPO] Rank {extended_rank}: Participated in state_dict gather for step {step}")
        
        # Only rank 0 waits for acknowledgment (actor will send it after collecting all shards)
        if extended_rank == 0:
            print(f"[GRPO] Step {step}: Waiting for ack from actor...")
            ack = recv_object(src_rank=7, tag=step*10+100)  # Use offset tag for ack
            print(f"[GRPO] Step {step}: Got ack: {ack}")
            assert ack["status"] == "OK", "Actor failed to update weights"
            
            # Request batch generation
            print(f"[GRPO] Step {step}: Requesting batch from actor...")
            send_object({
                "type": "GENERATE", 
                "num_rollouts": num_rollouts,
                "num_prompts": num_prompts,
                "max_gen_tokens": max_gen_tokens
            }, dst_rank=7, tag=step*10+2)
            print(f"[GRPO] Step {step}: Sent batch request, waiting for batch...")
            
            # Receive batch using tensor communication
            print(f"[GRPO] Step {step}: Receiving batch from actor...")
            
            # Receive shapes and metadata first
            batch_info = recv_object(src_rank=7, tag=step*10+3)
            shapes = batch_info["shapes"]
            metadata = batch_info["metadata"]
            
            # Receive tensors
            device = torch.cuda.current_device()
            sequences = torch.empty(shapes["sequences"], dtype=torch.long, device=device)
            rewards = torch.empty(shapes["rewards"], dtype=torch.float32, device=device)
            gen_logps = torch.empty(shapes["gen_logps"], dtype=torch.float32, device=device)
            ref_logps = torch.empty(shapes["ref_logps"], dtype=torch.float32, device=device)
            
            dist.recv(sequences, src=7, tag=step*10+10)
            dist.recv(rewards, src=7, tag=step*10+11)
            dist.recv(gen_logps, src=7, tag=step*10+12)
            dist.recv(ref_logps, src=7, tag=step*10+13)
            
            # Reconstruct batch dict
            batch = {
                "sequences": sequences.cpu().tolist(),
                "rewards": rewards.cpu().tolist(),
                "gen_logps": gen_logps.cpu().tolist(),
                "ref_logps": ref_logps.cpu().tolist(),
                "prompt_lens": metadata["prompt_lens"],
                "prompts": metadata["prompts"],
                "answers": metadata["answers"],
                "ans_token_ids": metadata["ans_token_ids"],
                "gen_time": metadata["gen_time"]
            }
            
            print(f"[GRPO] Step {step}: Received batch from actor")
        else:
            print(f"[GRPO] Rank {extended_rank}: Waiting for batch broadcast in step {step}")
            batch = None

        # Broadcast batch to all training ranks using tensor communication
        device = torch.cuda.current_device()
        
        if extended_rank == 0:
            # Rank 0 converts batch to tensors and broadcasts
            sequences = torch.tensor(batch["sequences"], dtype=torch.long, device=device)
            rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
            gen_old = torch.tensor(batch["gen_logps"], dtype=torch.float32, device=device)
            ref_logps = torch.tensor(batch["ref_logps"], dtype=torch.float32, device=device)
            
            # Broadcast metadata using small object
            metadata = {
                "prompt_lens": batch["prompt_lens"],
                "ans_token_ids": batch["ans_token_ids"],
                "prompts": batch["prompts"]
            }
            obj_list = [metadata]
            dist.broadcast_object_list(obj_list, src=0, group=trainer_group)
            
            # Broadcast tensors
            dist.broadcast(sequences, src=0, group=trainer_group)
            dist.broadcast(rewards, src=0, group=trainer_group)
            dist.broadcast(gen_old, src=0, group=trainer_group)
            dist.broadcast(ref_logps, src=0, group=trainer_group)
            
        else:
            # Other ranks receive broadcasts
            # Receive metadata
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0, group=trainer_group)
            metadata = obj_list[0]
            
            # Create empty tensors with the right shapes (we'll get the shapes from rank 0's broadcast)
            # For now, use dummy tensors - the broadcast will resize them appropriately
            sequences = torch.empty(1, dtype=torch.long, device=device)
            rewards = torch.empty(1, dtype=torch.float32, device=device)
            gen_old = torch.empty(1, dtype=torch.float32, device=device)
            ref_logps = torch.empty(1, dtype=torch.float32, device=device)
            
            # Receive tensor broadcasts
            dist.broadcast(sequences, src=0, group=trainer_group)
            dist.broadcast(rewards, src=0, group=trainer_group)
            dist.broadcast(gen_old, src=0, group=trainer_group)
            dist.broadcast(ref_logps, src=0, group=trainer_group)
        
        # Convert to expected dtypes
        rewards = rewards.to(torch.bfloat16)
        gen_old = gen_old.to(torch.bfloat16)
        ref_logps = ref_logps.to(torch.bfloat16)
        
        prompt_lens = metadata["prompt_lens"]
        ans_token_ids = metadata["ans_token_ids"]

        # Compute advantages per prompt
        num_prompts_in_batch = len(metadata["prompts"])
        num_rollouts_per_prompt = len(rewards) // num_prompts_in_batch
        rewards_per_prompt = rewards.view(num_prompts_in_batch, num_rollouts_per_prompt)
        mean_per_prompt = rewards_per_prompt.mean(dim=1, keepdim=True)
        std_per_prompt = rewards_per_prompt.std(dim=1, keepdim=True)
        advantages_per_prompt = (rewards_per_prompt - mean_per_prompt) / (std_per_prompt + 1e-4)
        advantages = advantages_per_prompt.flatten()

        # Forward pass
        logits = model(sequences).logits
        full_logps = per_token_logps(logits[:, :-1, :], sequences[:, 1:])

        # Extract logps for generated tokens
        gen_len = gen_old.shape[1]
        curr_logps_list = []
        for i in range(full_logps.shape[0]):
            p_len = prompt_lens[i]
            ans_len = len(ans_token_ids[i])
            sliced_logps = full_logps[i, p_len - 1 : p_len - 1 + ans_len]
            padded_logps = F.pad(sliced_logps, (0, gen_len - ans_len), 'constant', 0)
            curr_logps_list.append(padded_logps)
        curr_logps = torch.stack(curr_logps_list, dim=0)

        # PPO loss computation
        ratios = torch.exp(curr_logps - gen_old.detach())
        clipped = torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
        advantages = advantages.unsqueeze(-1)
        pg_loss = -torch.min(ratios * advantages, clipped * advantages)

        # KL regularization
        kl = torch.exp(ref_logps - curr_logps) - (ref_logps - curr_logps) - 1
        per_tok_loss = pg_loss + beta * kl

        # Create mask for loss computation
        max_ans_len = max(len(x) for x in ans_token_ids)
        mask = torch.zeros(len(ans_token_ids), max_ans_len, device=device, dtype=torch.float32)
        for i, tokens in enumerate(ans_token_ids):
            mask[i, :len(tokens)] = 1.0

        loss = ((per_tok_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)).mean()
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()
        
        # Gradient accumulation: only step optimizer every gradient_accumulation_steps
        if step % gradient_accumulation_steps == 0:
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        else:
            grad_norm = None

        # Logging (rank 0 only)
        if extended_rank == 0:
            print(f"Step: {step:05d} | Loss: {loss.item():.4f} | Mean Reward: {rewards.mean().item():.3f}")
            
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/reward", rewards.mean().item(), step)
            writer.add_scalar("train/gen_time", batch['gen_time'], step)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm, step)
            
            # Additional metrics
            writer.add_scalar("train/rewards_per_prompt/mean", mean_per_prompt.mean().item(), step)
            writer.add_scalar("train/rewards_per_prompt/std", std_per_prompt.mean().item(), step)
            writer.add_scalar("train/ratios/mean", ratios.mean().item(), step)
            writer.add_scalar("train/ratios/std", ratios.std().item(), step)
            writer.add_scalar("train/kl/mean", kl.mean().item(), step)
            writer.add_scalar("train/kl/std", kl.std().item(), step)

    print(f"[GRPO] Rank {extended_rank} training completed")

    # Cleanup - send stop signal to actor (only rank 0)
    if extended_rank == 0:
        print(f"[GRPO] Rank {extended_rank}: Sending stop signal to actor...")
        send_object({"type": "STOP"}, dst_rank=7, tag=99999*10)
        print(f"[GRPO] Rank {extended_rank}: Sent stop signal to actor")
    
    # Proper cleanup
    print(f"[GRPO] Rank {extended_rank} cleaning up...")
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"[GRPO] Rank {extended_rank} cleanup warning: {e}")

if __name__ == "__main__":
    main()