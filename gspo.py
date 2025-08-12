"""
GSPO trainer script - runs on GPUs 0-6 with FSDP
Communicates with actor.py via torch.distributed
Group Sequence Policy Optimization - operates at sequence level instead of token level
"""

import os, random, time, json, argparse
from typing import List, Dict, Any
# REMOVED: import pickle

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
    parser.add_argument("--job_id", type=str, default=None, help="SLURM job ID for logging")
    
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

def send_metadata(meta: dict, dst_rank, tag=0):
    """
    Send a metadata dict using only tensor communication (as a JSON string tensor).
    """
    device = torch.cuda.current_device()
    meta_str = json.dumps(meta)
    meta_bytes = meta_str.encode('utf-8')
    meta_len = torch.tensor([len(meta_bytes)], dtype=torch.long, device=device)
    dist.send(meta_len, dst=dst_rank, tag=tag)
    meta_tensor = torch.tensor(list(meta_bytes), dtype=torch.uint8, device=device)
    dist.send(meta_tensor, dst=dst_rank, tag=tag+1)

def recv_metadata(src_rank, tag=0):
    """
    Receive a metadata dict using only tensor communication (as a JSON string tensor).
    """
    device = torch.cuda.current_device()
    meta_len = torch.empty([1], dtype=torch.long, device=device)
    dist.recv(meta_len, src=src_rank, tag=tag)
    meta_len = meta_len.item()
    meta_tensor = torch.empty([meta_len], dtype=torch.uint8, device=device)
    dist.recv(meta_tensor, src=src_rank, tag=tag+1)
    meta_bytes = bytes(meta_tensor.cpu().tolist())
    meta_str = meta_bytes.decode('utf-8')
    return json.loads(meta_str)

def broadcast_state_dict(state_dict, src_rank, dst_rank, tag_base=0):
    """
    Broadcast a state dict from src_rank to dst_rank (point-to-point, not collective).
    All ranks must call this function. Only src_rank provides state_dict, dst_rank receives.
    """
    device = torch.cuda.current_device()
    # 1. On src, create metadata: list of (name, shape, dtype)
    if dist.get_rank() == src_rank:
        keys = list(state_dict.keys())
        shapes = [list(t.shape) for t in state_dict.values()]
        dtypes = [str(t.dtype) for t in state_dict.values()]
        meta = {"keys": keys, "shapes": shapes, "dtypes": dtypes}
        send_metadata(meta, dst_rank=dst_rank, tag=tag_base)
    elif dist.get_rank() == dst_rank:
        meta = recv_metadata(src_rank=src_rank, tag=tag_base)
    else:
        return None  # Only src and dst participate
    # 2. Broadcast each tensor in order
    tensors = []
    for i, (name, shape, dtype_str) in enumerate(zip(meta["keys"], meta["shapes"], meta["dtypes"])):
        dtype = getattr(torch, dtype_str.split('.')[-1])
        if dist.get_rank() == src_rank:
            tensor = state_dict[name].to(device)
            dist.send(tensor, dst=dst_rank, tag=tag_base+10+i)
        elif dist.get_rank() == dst_rank:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            dist.recv(tensor, src=src_rank, tag=tag_base+10+i)
            tensors.append(tensor)
    if dist.get_rank() == dst_rank:
        state_dict = {k: t for k, t in zip(meta["keys"], tensors)}
        return state_dict
    return None

def send_batch(batch, dst_rank, tag=0):
    device = torch.cuda.current_device()
    # Convert lists to tensors and send them
    sequences = torch.tensor(batch["sequences"], dtype=torch.long, device=device)
    rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
    gen_logps = torch.tensor(batch["gen_logps"], dtype=torch.float32, device=device)
    ref_logps = torch.tensor(batch["ref_logps"], dtype=torch.float32, device=device)
    # Send tensor shapes first as metadata
    meta = {
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
    }
    send_metadata(meta, dst_rank=dst_rank, tag=tag)
    dist.send(sequences, dst=dst_rank, tag=tag+10)
    dist.send(rewards, dst=dst_rank, tag=tag+11)
    dist.send(gen_logps, dst=dst_rank, tag=tag+12)
    dist.send(ref_logps, dst=dst_rank, tag=tag+13)

def recv_batch(src_rank, tag=0):
    device = torch.cuda.current_device()
    meta = recv_metadata(src_rank=src_rank, tag=tag)
    shapes = meta["shapes"]
    metadata = meta["metadata"]
    sequences = torch.empty(shapes["sequences"], dtype=torch.long, device=device)
    rewards = torch.empty(shapes["rewards"], dtype=torch.float32, device=device)
    gen_logps = torch.empty(shapes["gen_logps"], dtype=torch.float32, device=device)
    ref_logps = torch.empty(shapes["ref_logps"], dtype=torch.float32, device=device)
    dist.recv(sequences, src=src_rank, tag=tag+10)
    dist.recv(rewards, src=src_rank, tag=tag+11)
    dist.recv(gen_logps, src=src_rank, tag=tag+12)
    dist.recv(ref_logps, src=src_rank, tag=tag+13)
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
            print(f"[ACTOR] Step {step}: Waiting for message from rank 0...")
            
            # First check if this is a STOP signal
            try:
                # Try to receive a control message first (could be STOP or state dict metadata)
                msg = recv_metadata(src_rank=0, tag=step*100)
                if msg.get("type") == "STOP":
                    print(f"[ACTOR] Received STOP signal, shutting down")
                    return
                elif msg.get("type") == "STATE_DICT":
                    print(f"[ACTOR] Step {step}: Receiving state dict from rank 0...")
                    # Continue with state dict reception
                else:
                    print(f"[ACTOR] Warning: Unknown message type: {msg}")
                    continue
            except Exception as e:
                print(f"[ACTOR] Error receiving control message: {e}")
                break
            
            # Receive state dict from rank 0
            state_dict = broadcast_state_dict(None, src_rank=0, dst_rank=actor_rank, tag_base=step*100+10)
            if state_dict is None:
                continue
            print(f"[ACTOR] Step {step}: Received all tensors, loading into model...")
            model_gen.load_state_dict(state_dict)
            print(f"[ACTOR] Step {step}: Weights updated")
            # Send acknowledgment to rank 0
            ack = {"status": "OK"}
            send_metadata(ack, dst_rank=0, tag=step*100+99)
            print(f"[ACTOR] Step {step}: Sent ack to rank 0")
            
            # Wait for generation request
            print(f"[ACTOR] Step {step}: Waiting for generation request...")
            gen_msg = recv_metadata(src_rank=0, tag=step*100+5)
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
            send_metadata({
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
            }, dst_rank=0, tag=step*100+20)
            
            # Send tensors
            dist.send(sequences, dst=0, tag=step*100+30)
            dist.send(rewards, dst=0, tag=step*100+31) 
            dist.send(gen_logps, dst=0, tag=step*100+32)
            dist.send(ref_logps, dst=0, tag=step*100+33)
            
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
    print(f"[GSPO] Setting up environment for distributed training")
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    print(f"[GSPO] Environment set: MASTER_ADDR={os.environ['MASTER_ADDR']} MASTER_PORT={os.environ['MASTER_PORT']}")
    
    # Initialize distributed training directly with extended world (ranks 0-7, where 7 is the actor)
    print(f"[GSPO] Initializing distributed training for extended world (ranks 0-7)")
    
    # Get local rank from environment (set by deepspeed launcher)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = 8  # 7 trainers + 1 actor
    
    torch.cuda.set_device(local_rank)
    
    print(f"[GSPO] Initializing process group: rank={global_rank}, world_size={world_size}")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        world_size=world_size,
        rank=global_rank
    )
    print(f"[GSPO] Process group initialized: rank {global_rank}/{world_size}")
    
    # Use global_rank as extended_rank for compatibility
    extended_rank = global_rank
    extended_world_size = world_size
    
    # Rank 7 acts as the actor, others as trainers
    if extended_rank == 7:
        print(f"[GSPO] Rank {extended_rank} will act as ACTOR")
        run_actor_role(extended_rank, extended_world_size, model_path, task, args.master_addr, args.master_port)
        return
    else:
        print(f"[GSPO] Rank {extended_rank} will act as TRAINER")
        
    # Create a separate process group for training ranks only (0-6)
    trainer_ranks = list(range(7))  # [0, 1, 2, 3, 4, 5, 6]
    trainer_group = dist.new_group(trainer_ranks)
    print(f"[GSPO] Created trainer process group with ranks {trainer_ranks}")
    
    print(f"[GSPO] Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model on CPU first, then move to FSDP
    print(f"[GSPO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    
    # Determine the transformer layer class for wrapping policy
    # For Qwen2.5, this should be Qwen2DecoderLayer
    try:
        from transformers.models.qwen2 import Qwen2DecoderLayer
        transformer_layer_cls = Qwen2DecoderLayer
    except ImportError:
        # Fallback: wrap every module 
        transformer_layer_cls = torch.nn.Module
        print(f"[GSPO] Warning: Could not import Qwen2DecoderLayer, using generic wrapping")
    
    # Wrap model with FSDP (using trainer group only)
    print(f"[GSPO] Initializing FSDP...")
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
    print(f"[GSPO] Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    print(f'[GSPO] Rank {extended_rank} FSDP model and optimizer created.')

    if extended_rank == 0:
        print(f"[GSPO] Creating SummaryWriter and logging args")
        logdir = f"runs/gspo_{args.task}_{args.job_id}" if args.job_id is not None else f"runs/gspo_{args.task}_{time.strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(logdir)
        # Log each CLI arg as a separate text entry
        for k, v in vars(args).items():
            writer.add_text(f"args/{k}", str(v))
        # also log the task
        

    # Wait for actor to be ready
    print(f"[GSPO] Rank {extended_rank} waiting for actor to be ready (barrier)...")
    dist.barrier()  # Actor will join this barrier when ready
    print(f"[GSPO] Rank {extended_rank} actor is ready, starting training loop")

    # Training loop
    for step in trange(1, args.steps + 1, desc="Training Steps", disable=(extended_rank != 0)):
        
        # Only rank 0 gets the full state dict and sends tensors to actor
        if extended_rank == 0:
            print(f"[GSPO] Rank {extended_rank}: Getting full state_dict for step {step}...")
            # Use full state dict on rank 0 - FSDP will gather from all ranks
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                full_state_dict = model.state_dict()
            print(f"[GSPO] Rank {extended_rank}: Got full state_dict for step {step}")
            
            print(f"[GSPO] Rank {extended_rank}: Sending state dict to actor...")
            
            # Send control message first
            send_metadata({"type": "STATE_DICT"}, dst_rank=7, tag=step*100)
            
            # Send state dict to actor (rank 7)
            broadcast_state_dict(full_state_dict, src_rank=0, dst_rank=7, tag_base=step*100+10)
            
            print(f"[GSPO] Rank {extended_rank}: Sent state dict to actor")
        else:
            # Other ranks just participate in the collective gather
            print(f"[GSPO] Rank {extended_rank}: Participating in state_dict gather for step {step}")
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                model.state_dict()  # Participate in collective but don't use result
            print(f"[GSPO] Rank {extended_rank}: Participated in state_dict gather for step {step}")
        
        # Only rank 0 waits for acknowledgment (actor will send it after collecting all shards)
        if extended_rank == 0:
            print(f"[GSPO] Step {step}: Waiting for ack from actor...")
            ack = recv_metadata(src_rank=7, tag=step*100+99)  # Use offset tag for ack
            print(f"[GSPO] Step {step}: Got ack: {ack}")
            assert ack["status"] == "OK", "Actor failed to update weights"
            
            # Request batch generation
            print(f"[GSPO] Step {step}: Requesting batch from actor...")
            gen_msg = {
                "type": "GENERATE", 
                "num_rollouts": num_rollouts,
                "num_prompts": num_prompts,
                "max_gen_tokens": max_gen_tokens
            }
            send_metadata(gen_msg, dst_rank=7, tag=step*100+5)
            print(f"[GSPO] Step {step}: Sent batch request, waiting for batch...")
            
            # Receive batch using tensor communication
            print(f"[GSPO] Step {step}: Receiving batch from actor...")
            
            # Receive shapes and metadata first
            batch_info = recv_metadata(src_rank=7, tag=step*100+20)
            shapes = batch_info["shapes"]
            metadata = batch_info["metadata"]
            
            # Receive tensors
            device = torch.cuda.current_device()
            sequences = torch.empty(shapes["sequences"], dtype=torch.long, device=device)
            rewards = torch.empty(shapes["rewards"], dtype=torch.float32, device=device)
            gen_logps = torch.empty(shapes["gen_logps"], dtype=torch.float32, device=device)
            ref_logps = torch.empty(shapes["ref_logps"], dtype=torch.float32, device=device)
            
            dist.recv(sequences, src=7, tag=step*100+30)
            dist.recv(rewards, src=7, tag=step*100+31)
            dist.recv(gen_logps, src=7, tag=step*100+32)
            dist.recv(ref_logps, src=7, tag=step*100+33)
            
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
            
            print(f"[GSPO] Step {step}: Received batch from actor")
        else:
            print(f"[GSPO] Rank {extended_rank}: Waiting for batch broadcast in step {step}")
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
                "prompts": batch["prompts"],
                "shapes": {
                    "sequences": list(sequences.shape),
                    "rewards": list(rewards.shape),
                    "gen_old": list(gen_old.shape),
                    "ref_logps": list(ref_logps.shape)
                }
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
            # Receive metadata first to get shapes
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0, group=trainer_group)
            metadata = obj_list[0]
            shapes = metadata["shapes"]
            
            # Create tensors with the correct shapes from metadata
            sequences = torch.empty(shapes["sequences"], dtype=torch.long, device=device)
            rewards = torch.empty(shapes["rewards"], dtype=torch.float32, device=device)
            gen_old = torch.empty(shapes["gen_old"], dtype=torch.float32, device=device)
            ref_logps = torch.empty(shapes["ref_logps"], dtype=torch.float32, device=device)
            
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

        # Compute advantages per prompt (same as GRPO)
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

        # GSPO: Compute sequence-level log probabilities instead of token-level
        curr_seq_logps = []
        gen_seq_logps = []
        ref_seq_logps = []
        seq_lengths = []
        
        for i in range(full_logps.shape[0]):
            p_len = prompt_lens[i]
            ans_len = len(ans_token_ids[i])
            
            # Extract log probabilities for actual generated tokens only (no padding)
            curr_slice = full_logps[i, p_len - 1 : p_len - 1 + ans_len]
            gen_slice = gen_old[i, :ans_len]  # Use actual length, not padded
            ref_slice = ref_logps[i, :ans_len]  # Use actual length, not padded
            
            # Sum log probabilities to get sequence-level probabilities
            curr_seq_logp = curr_slice.sum()
            gen_seq_logp = gen_slice.sum()
            ref_seq_logp = ref_slice.sum()
            
            curr_seq_logps.append(curr_seq_logp)
            gen_seq_logps.append(gen_seq_logp)
            ref_seq_logps.append(ref_seq_logp)
            seq_lengths.append(ans_len)
        
        # --- GSPO: Compute length-normalized log-ratios and apply fixes ---
        curr_seq_logps = torch.stack(curr_seq_logps).to(torch.float32)
        gen_seq_logps = torch.stack(gen_seq_logps).to(torch.float32)
        ref_seq_logps = torch.stack(ref_seq_logps).to(torch.float32)
        seq_lengths = torch.tensor(seq_lengths, device=device, dtype=torch.float32)

        # 1. Length-normalized log-ratios
        log_r = (curr_seq_logps - gen_seq_logps.detach()) / seq_lengths
        log_r_ref = (ref_seq_logps.detach() - curr_seq_logps) / seq_lengths

        # 2. Importance ratio and clipping (on normalized ratio)
        seq_ratios = torch.exp(log_r)
        seq_clipped = torch.clamp(seq_ratios, 1 - clip_param, 1 + clip_param)
        seq_pg_loss = -torch.min(seq_ratios * advantages, seq_clipped * advantages)

        # 3. KL penalty (on normalized log-ratio)
        seq_kl = torch.exp(log_r_ref) - log_r_ref - 1

        # 4. Final sequence-level loss (no further division by seq_lengths)
        seq_loss = seq_pg_loss + beta * seq_kl
        loss = seq_loss.mean()

        # 5. Scale loss for gradient accumulation
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

            # Scalar logs
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/reward", rewards.mean().item(), step)
            writer.add_scalar("train/gen_time", batch['gen_time'], step)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm, step)
            writer.add_scalar("train/rewards_per_prompt/mean", mean_per_prompt.mean().item(), step)
            writer.add_scalar("train/rewards_per_prompt/std", std_per_prompt.mean().item(), step)
            writer.add_scalar("train/seq_ratios/mean", seq_ratios.mean().item(), step)
            writer.add_scalar("train/seq_ratios/std", seq_ratios.std().item(), step)
            writer.add_scalar("train/log_r/mean", log_r.mean().item(), step)
            writer.add_scalar("train/seq_kl/mean", seq_kl.mean().item(), step)
            writer.add_scalar("train/seq_kl/std", seq_kl.std().item(), step)
            writer.add_scalar("train/seq_pg_loss/mean", seq_pg_loss.mean().item(), step)
            writer.add_scalar("train/seq_pg_loss/std", seq_pg_loss.std().item(), step)
            writer.add_scalar("train/advantages_per_prompt/mean", advantages_per_prompt.mean().item(), step)
            writer.add_scalar("train/advantages_per_prompt/std", advantages_per_prompt.std().item(), step)
            writer.add_scalar("train/seq_lengths/mean", seq_lengths.mean().item(), step)
            writer.add_scalar("train/seq_lengths/std", seq_lengths.std().item(), step)

            # Completion length stats
            completion_lengths = torch.tensor([len(x) for x in ans_token_ids], dtype=torch.float32)
            writer.add_scalar("gen/completion/mean_length", completion_lengths.mean().item(), step)
            writer.add_scalar("gen/completion/min_length", completion_lengths.min().item(), step)
            writer.add_scalar("gen/completion/max_length", completion_lengths.max().item(), step)

            # Reward stats
            rewards_tensor = torch.tensor(batch['rewards'])
            writer.add_scalar("gen/rewards_mean", rewards_tensor.mean().item(), step)
            writer.add_scalar("gen/rewards_std", rewards_tensor.std().item(), step)
            writer.add_scalar("gen/rewards_min", rewards_tensor.min().item(), step)
            writer.add_scalar("gen/rewards_max", rewards_tensor.max().item(), step)

            # Table logging
            prompts = batch['prompts']
            answers = batch['answers']
            num_prompts_in_batch = len(prompts)
            num_rollouts_per_prompt = len(answers) // num_prompts_in_batch
            rewards_per_prompt_log = rewards_tensor.view(num_prompts_in_batch, num_rollouts_per_prompt)
            mean_per_prompt_log = rewards_per_prompt_log.mean(dim=1, keepdim=True)
            std_per_prompt_log = rewards_per_prompt_log.std(dim=1, keepdim=True)
            advantages_for_logging = (rewards_per_prompt_log - mean_per_prompt_log) / (std_per_prompt_log + 1e-4)
            advantages_for_logging = advantages_for_logging.flatten()

            for p_idx in range(num_prompts_in_batch):
                prompt_text = prompts[p_idx]
                S = f"Prompt: {prompt_text}<br><br>"
                for r_idx in range(num_rollouts_per_prompt):
                    ans_idx = p_idx * num_rollouts_per_prompt + r_idx
                    answer_text = answers[ans_idx]
                    reward_val = rewards_tensor[ans_idx].item()
                    adv_val = advantages_for_logging[ans_idx].item()
                    S += f"Answer: {answer_text}<br><br>"
                    S += f"Reward: {reward_val}<br><br>"
                    S += f"Advantage: {adv_val}<br><br>"
                writer.add_text("generations", S, step)

    print(f"[GSPO] Rank {extended_rank} training completed")

    # Cleanup - send stop signal to actor (only rank 0)
    if extended_rank == 0:
        print(f"[GSPO] Rank {extended_rank}: Sending stop signal to actor...")
        send_metadata({"type": "STOP"}, dst_rank=7, tag=(args.steps+1)*100)
        print(f"[GSPO] Rank {extended_rank}: Sent stop signal to actor")
    
    # Proper cleanup
    print(f"[GSPO] Rank {extended_rank} cleaning up...")
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"[GSPO] Rank {extended_rank} cleanup warning: {e}")

if __name__ == "__main__":
    main()
