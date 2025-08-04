"""
GRPO actor script - runs on GPU 7 for inference
Communicates with grpo.py via torch.distributed
"""

import os, random, time, json, argparse
from typing import List, Dict, Any
import pickle

# Delay heavy imports until after environment is set in main
torch = None
AutoTokenizer = None
AutoModelForCausalLM = None
get_task = None

def _lazy_imports():
    global torch, AutoTokenizer, AutoModelForCausalLM, get_task, dist
    if torch is None:
        import torch
        import torch.distributed as dist
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from tasks import get_task
        globals()['torch'] = torch
        globals()['dist'] = dist
        globals()['AutoTokenizer'] = AutoTokenizer
        globals()['AutoModelForCausalLM'] = AutoModelForCausalLM
        globals()['get_task'] = get_task

# ----------------------------------------------------------------------------------
# CLI Arguments
# ----------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-instruct")
    parser.add_argument("--task", type=str, default="gsm8k")
    
    # Coordination parameters
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="29500")
    parser.add_argument("--actor_gpu", type=int, default=7, help="Global GPU id for actor")
    
    return parser.parse_args()

args = get_args()

# ----------------------------------------------------------------------------------
# Communication utilities
# ----------------------------------------------------------------------------------

def send_object(obj, dst_rank, tag=0):
    """Send a Python object to another rank via torch.distributed"""
    data = pickle.dumps(obj)
    
    # Send size first
    size_tensor = torch.tensor([len(data)], dtype=torch.long, device=torch.cuda.current_device())
    dist.send(size_tensor, dst=dst_rank, tag=tag)
    
    # Send data
    data_tensor = torch.tensor(list(data), dtype=torch.uint8, device=torch.cuda.current_device())
    dist.send(data_tensor, dst=dst_rank, tag=tag)

def recv_object(src_rank, tag=0):
    """Receive a Python object from another rank via torch.distributed"""
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

# ----------------------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------------------

def main():
    # Set GPU visibility before importing CUDA
    print(f"[ACTOR] CUDA_VISIBLE_DEVICES is {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    
    print(f"[ACTOR] Setting MASTER_ADDR and MASTER_PORT")
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    print(f"[ACTOR] MASTER_ADDR={os.environ['MASTER_ADDR']} MASTER_PORT={os.environ['MASTER_PORT']}")
    
    # Initialize distributed as rank 7 in world size 8
    actor_rank = 7
    world_size = 8
    
    # Set device (GPU 7 becomes cuda:0 in this process)
    _lazy_imports()
    print(f"[ACTOR] Imports loaded, torch version: {torch.__version__}")
    print(f"[ACTOR] Setting torch.cuda.set_device(0)")
    torch.cuda.set_device(0)
    actor_device = torch.device("cuda:0")
    print(f"[ACTOR] torch.cuda.set_device done, actor_device={actor_device}")
    
    print(f"[ACTOR] Initializing process group: rank={actor_rank}, world_size={world_size}")
    print(f"[ACTOR] Waiting 10 seconds for GRPO processes to set up process group...")
    time.sleep(10)  # Give GRPO processes time to destroy and recreate their process group
    
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        world_size=world_size,
        rank=actor_rank
    )
    print(f"[ACTOR] Process group initialized")
    
    print(f"[ACTOR] Loading tokenizer and models")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"[ACTOR] Loading models...")
    model_gen = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(actor_device)
    model_ref = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(actor_device)
    model_gen.eval()
    model_ref.eval()
    print(f"[ACTOR] Models loaded")
    
    print(f"[ACTOR] Loading task...")
    task_items, reward_fn = get_task(args.task)
    print(f"[ACTOR] Task '{args.task}' loaded with {len(task_items)} items")
    
    # System prompt
    system_prompt = (
        "You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags."
    )
    
    # Signal readiness to training ranks
    print(f"[ACTOR] Signaling readiness (barrier)...")
    dist.barrier()  # Join the barrier that grpo.py is waiting on
    print(f"[ACTOR] Ready for training loop")
    
    step = 0
    try:
        while True:
            step += 1
            print(f"[ACTOR] Step {step}: Waiting for message from rank 0...")
            
            # Receive message from rank 0
            msg = recv_object(src_rank=0, tag=step*10)
            print(f"[ACTOR] Step {step}: Received message: {msg}")
            msg_type = msg["type"]
            
            if msg_type == "STOP":
                print(f"[ACTOR] Received STOP signal, shutting down")
                break
            elif msg_type == "UPDATE_WEIGHTS":
                print(f"[ACTOR] Step {step}: Updating weights...")
                state_dict = msg["state_dict"]
                # Move state dict to GPU
                state_dict_gpu = {k: v.to(actor_device) for k, v in state_dict.items()}
                model_gen.load_state_dict(state_dict_gpu)
                print(f"[ACTOR] Step {step}: Weights updated")
                
                # Send acknowledgment
                send_object({"status": "OK"}, dst_rank=0, tag=step*10+1)
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
                    num_prompts, num_rollouts, max_gen_tokens, system_prompt, actor_device
                )
                
                gen_time = time.time() - tic
                batch["gen_time"] = gen_time
                
                print(f"[ACTOR] Step {step}: Generated batch in {gen_time:.2f}s, sending to rank 0...")
                
                # Send batch back to rank 0
                send_object(batch, dst_rank=0, tag=step*10+3)
                print(f"[ACTOR] Step {step}: Batch sent")
                
            else:
                print(f"[ACTOR] Step {step}: Unknown message type: {msg_type}")
                raise RuntimeError(f"Unknown message type: {msg_type}")
                
    except Exception as e:
        print(f"[ACTOR] Error in training loop: {e}")
        raise
    
    print(f"[ACTOR] Actor shutting down")
    
    # Proper cleanup
    print(f"[ACTOR] Cleaning up...")
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"[ACTOR] Cleanup warning: {e}")

def generate_batch(model_gen, model_ref, tokenizer, reward_fn, task_items, 
                  num_prompts, num_rollouts, max_gen_tokens, system_prompt, device):
    """Generate a batch of rollouts"""
    
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

if __name__ == "__main__":
    main()