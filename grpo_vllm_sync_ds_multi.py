"""
Distributed synchronous GRPO with:
  • DeepSpeed learner on N GPUs (data-parallel ranks)
  • vLLM actor on *one* dedicated GPU that is **not** part of the DeepSpeed
    visibility set.  Example launch command:

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
deepspeed --include localhost:0,1,2,3,4,5,6 \
          grpo_vllm_sync_ds_multi.py \
          --actor_gpu 7  # Note: do NOT pass --num_gpus when using --include

The learner (rank 0) runs the actor in a separate process, passing weights and
receiving batches every step.  Other ranks receive the batch via
`dist.broadcast_object_list` and compute the same loss.

NOTE 1:  All ranks compute the *same* loss; therefore the effective batch size
        is not multiplied by world_size – gradients will be averaged by
        DistributedDataParallel / ZeRO.

NOTE 2:  Because the actor GPU is outside `CUDA_VISIBLE_DEVICES`, the child
        process sets its own environment variable before importing CUDA libs.
"""

import os, random, time, json, argparse
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter

import deepspeed
from tqdm import trange

from tasks import get_task

# ----------------------------------------------------------------------------------
# Hyper-parameters (can also be overridden by CLI args)
# ----------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_gpu", type=int, default=1, help="Global GPU id reserved for vLLM actor")
    # train_gpus remains for completeness but is not used by the script logic.
    parser.add_argument("--train_gpus", type=int, default=7, help="Number of GPUs allocated to DeepSpeed learner")
    # Deepspeed launcher passes --local_rank to every process; we must accept it.
    parser.add_argument("--local_rank", type=int, default=-1, help="(internal) local rank passed by DeepSpeed")
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
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

args = get_args()

# Static params -----------------------------------------------------------

task = args.task
model_path = args.model_path
num_rollouts = args.num_rollouts
num_prompts = args.num_prompts
gradient_accumulation_steps = args.gradient_accumulation_steps
max_gen_tokens = args.max_gen_tokens
beta = args.beta
clip_param = args.clip_param
lr = args.lr
num_train_devices = 7

# Your logic: num_rollouts * num_prompts = train_batch_size * grad_accum
# train_batch_size here is train_micro_batch_size_per_gpu
train_mini_batch_size= (num_prompts * num_rollouts) // gradient_accumulation_steps
train_micro_batch_size_per_gpu = train_mini_batch_size // num_train_devices

if (num_prompts * num_rollouts) % gradient_accumulation_steps != 0:
    raise ValueError("num_prompts * num_rollouts must be divisible by gradient_accumulation_steps")

#if train_micro_batch_size_per_gpu % num_rollouts != 0:
#    raise ValueError("train_micro_batch_size_per_gpu must be a multiple of num_rollouts")


# DeepSpeed runtime config ------------------------------------------------
DS_CONFIG: Dict[str, Any] = {
    "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "optimizer": {"type": "AdamW", "params": {"lr": lr}},
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "offload_optimizer": {"device": "cpu"},
    },
    "gradient_clipping": 1.0,  # Added gradient clipping
}

# ----------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------

def per_token_logps(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return torch.gather(logp, -1, ids.unsqueeze(-1)).squeeze(-1)

# ----------------------------------------------------------------------------------
# Actor process (runs only once, launched by rank-0)
# ----------------------------------------------------------------------------------

def actor_process(actor_gpu: int, req_q: mp.Queue, resp_q: mp.Queue, num_rollouts: int, num_prompts: int, max_gen_tokens: int):
    # make chosen GPU the only visible one for this process BEFORE importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(actor_gpu)
    import torch  # re-import inside new context

    # The actor is on a single GPU, so it becomes cuda:0 inside this process
    actor_device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Use separate models for generation and reference
    model_gen = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(actor_device)
    model_ref = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(actor_device)
    model_gen.eval()
    model_ref.eval()

    # Inform the learner that the actor (vLLM thread) is fully initialized and
    # ready to accept requests.  This prevents the training loop from starting
    # before the heavyweight model loading has finished.
    resp_q.put({"status": "READY"})

    task_items, reward_fn = get_task(task)

    system_prompt = (
        """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags."""
    )

    while True:
        msg = req_q.get()
        mtype = msg["type"]
        if mtype == "STOP":
            break
        elif mtype == "UPDATE_WEIGHTS":
            print('[Actor] Updating weights...')
            model_gen.load_state_dict(msg["state_dict"])
            print('[Actor] Weights updated.')
            resp_q.put({"status": "OK"})
        elif mtype == "GENERATE":
            print('[Actor] Generating new batch...')
            tic = time.time()
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

                tip_text = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                inputs = tokenizer(tip_text, return_tensors="pt").to(actor_device)
                prompt_len = inputs.input_ids.shape[1]
                
                # Append prompt_len for each rollout
                all_prompt_lens.extend([prompt_len] * num_rollouts)

                # Generate sequences
                full_sequences_ids = model_gen.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9,
                    max_new_tokens=max_gen_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=num_rollouts,
                )

                # Decode answers (completion only)
                ans_token_ids = full_sequences_ids[:, prompt_len:]
                answers = tokenizer.batch_decode(ans_token_ids, skip_special_tokens=True)

                rewards = reward_fn(answers, items=[item] * len(answers))

                def get_logprobs_for_generated(model, full_ids, p_len):
                    with torch.no_grad():
                        logits = model(full_ids).logits
                    log_softmax = torch.log_softmax(logits, dim=-1)
                    log_probs = torch.gather(log_softmax[:, :-1, :], -1, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                    return log_probs[:, p_len - 1 :]

                gen_logps = get_logprobs_for_generated(model_gen, full_sequences_ids, prompt_len)
                ref_logps = get_logprobs_for_generated(model_ref, full_sequences_ids, prompt_len)

                all_sequences.extend(full_sequences_ids.tolist())
                all_rewards.extend(rewards)
                all_gen_logps.extend(gen_logps.cpu().tolist())
                all_ref_logps.extend(ref_logps.cpu().tolist())
                all_answers.extend(answers)
                all_ans_token_ids.extend(ans_token_ids.cpu().tolist())

            gen_time = time.time() - tic

            # Pad sequences to the same length for tensor conversion on the learner side.
            max_seq_len = max(len(s) for s in all_sequences)
            for seq in all_sequences:
                seq.extend([tokenizer.pad_token_id] * (max_seq_len - len(seq)))
            
            # Also pad the logp tensors to a common generation length
            max_gen_len = max(len(l) for l in all_gen_logps)
            for l in all_gen_logps:
                l.extend([0.0] * (max_gen_len - len(l)))
            for l in all_ref_logps:
                l.extend([0.0] * (max_gen_len - len(l)))

            resp_q.put(
                {
                    "sequences": all_sequences,
                    "rewards": all_rewards,
                    "gen_logps": all_gen_logps,
                    "ref_logps": all_ref_logps,
                    "prompt_lens": all_prompt_lens,
                    "gen_time": gen_time,
                    "prompts": all_prompts,
                    "answers": all_answers,
                    "ans_token_ids": all_ans_token_ids,
                }
            )
        else:
            raise RuntimeError(f"Unknown msg type {mtype}")

# ----------------------------------------------------------------------------------
# Main learner function (executed by every rank)
# ----------------------------------------------------------------------------------

def main():
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # print some logs
    print(f"[Rank {rank}/{world_size}] Initialized. Local rank: {local_rank}")

    # Queues only on rank-0
    if rank == 0:
        mp.set_start_method("spawn", force=True)
        req_q, resp_q = mp.Queue(), mp.Queue()
        actor = mp.Process(
            target=actor_process,
            args=(args.actor_gpu, req_q, resp_q, num_rollouts, num_prompts, max_gen_tokens),
        )
        actor.start()
    else:
        req_q = resp_q = None  # type: ignore

    # Tokenizer & model (same on all ranks)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    engine, optimizer, _, _ = deepspeed.initialize(
        config=DS_CONFIG,
        model=model,
        model_parameters=model.parameters(),
    )

    # print some logs
    print(f'[Rank {rank}] DeepSpeed engine created.')

    if rank == 0:
        writer = SummaryWriter(f"runs/sync_multi_{time.strftime('%Y%m%d-%H%M%S')}")
        # log the args as text
        writer.add_text("args/task", args.task)
        writer.add_text("args/model_path", args.model_path)
        writer.add_text("args/num_rollouts", str(args.num_rollouts))
        writer.add_text("args/num_prompts", str(args.num_prompts))
        writer.add_text("args/gradient_accumulation_steps", str(args.gradient_accumulation_steps))
        writer.add_text("args/max_gen_tokens", str(args.max_gen_tokens))
        writer.add_text("args/beta", str(args.beta))
        writer.add_text("args/clip_param", str(args.clip_param))
        writer.add_text("args/lr", str(args.lr))
        writer.add_text("args/train_gpus", str(args.train_gpus))
        writer.add_text("args/actor_gpu", str(args.actor_gpu))
        writer.add_text("args/steps", str(args.steps))

    # ------------------------------------------------------------------
    # Wait until the actor process has finished loading its models and
    # signals readiness, then synchronize all ranks so the training loop
    # only starts once everything is in place.
    # ------------------------------------------------------------------
    if rank == 0:
        ready_msg = resp_q.get()  # blocks until actor sends a READY status
        if ready_msg.get("status") != "READY":
            raise RuntimeError("Actor failed to initialize properly")

    # Ensure every rank waits until the actor is ready
    dist.barrier()

    # Start training loop
    for step in trange(1, args.steps + 1, desc="Training Steps"):
        # print(f"step {step}, rank {rank}, local_rank {local_rank}")
        # ---------------- rank-0 interacts with actor ---------------------
        if rank == 0:
            # push weights
            state_dict_cpu = {k: v.to("cpu") for k, v in engine.module.state_dict().items()}
            req_q.put({"type": "UPDATE_WEIGHTS", "state_dict": state_dict_cpu})
            print(f'[Rank {rank}] Sending weights to actor.')
            resp_q.get()
            # request batch
            req_q.put({"type": "GENERATE"})
            print(f'[Rank {rank}] Requesting new batch from actor.')
            batch = resp_q.get()
            print(f'[Rank {rank}] Received batch from actor.')
        else:
            batch = None

        # broadcast batch dict to all ranks
        obj_list = [batch]
        dist.broadcast_object_list(obj_list, src=0)
        if rank != 0:
            batch = obj_list[0]

        # ---------------- compute loss locally ---------------------------
        sequences = torch.tensor(batch["sequences"], dtype=torch.long, device=engine.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.bfloat16, device=engine.device)
        gen_old = torch.tensor(batch["gen_logps"], dtype=torch.bfloat16, device=engine.device)
        ref_logps = torch.tensor(batch["ref_logps"], dtype=torch.bfloat16, device=engine.device)
        prompt_lens = batch["prompt_lens"]
        ans_token_ids = batch["ans_token_ids"]

        # Group rewards by prompt to calculate advantages
        num_prompts_in_batch = len(batch["prompts"])
        num_rollouts_per_prompt = len(rewards) // num_prompts_in_batch
        rewards_per_prompt = rewards.view(num_prompts_in_batch, num_rollouts_per_prompt)
        mean_per_prompt = rewards_per_prompt.mean(dim=1, keepdim=True)
        std_per_prompt = rewards_per_prompt.std(dim=1, keepdim=True)
        advantages_per_prompt = (rewards_per_prompt - mean_per_prompt) / (std_per_prompt + 1e-4)
        advantages = advantages_per_prompt.flatten()
        if rank == 0:
            writer.add_scalar("train/rewards_per_prompt/mean", mean_per_prompt.mean().item(), step)
            writer.add_scalar("train/rewards_per_prompt/std", std_per_prompt.mean().item(), step)
            writer.add_scalar("train/advantages_per_prompt/mean", advantages_per_prompt.mean().item(), step)
            writer.add_scalar("train/advantages_per_prompt/std", advantages_per_prompt.std().item(), step)


        logits = engine(sequences).logits
        full_logps = per_token_logps(logits[:, :-1, :], sequences[:, 1:])

        # Extract logps for the generated part of each sequence, handling variable prompt lengths
        gen_len = gen_old.shape[1]
        curr_logps_list = []
        for i in range(full_logps.shape[0]):
            p_len = prompt_lens[i]
            # logps for generated tokens start at index p_len-1
            # The length of generated part is the number of non-pad tokens in ans_token_ids
            ans_len = len(ans_token_ids[i])
            sliced_logps = full_logps[i, p_len - 1 : p_len - 1 + ans_len]
            
            # Pad to max generation length so we can stack them
            padded_logps = F.pad(sliced_logps, (0, gen_len - ans_len), 'constant', 0)
            curr_logps_list.append(padded_logps)
        curr_logps = torch.stack(curr_logps_list, dim=0)

        ratios = torch.exp(curr_logps - gen_old.detach())
        clipped = torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
        advantages = advantages.unsqueeze(-1)
        pg_loss = -torch.min(ratios * advantages, clipped * advantages)
        if rank == 0:
            writer.add_scalar("train/ratios/mean", ratios.mean().item(), step)
            writer.add_scalar("train/ratios/std", ratios.std().item(), step)
            writer.add_scalar("train/pg_loss/mean", pg_loss.mean().item(), step)
            writer.add_scalar("train/pg_loss/std", pg_loss.std().item(), step)

        kl = torch.exp(ref_logps - curr_logps) - (ref_logps - curr_logps) - 1
        per_tok_loss = pg_loss + beta * kl

        if rank == 0:
            writer.add_scalar("train/kl/mean", kl.mean().item(), step)
            writer.add_scalar("train/kl/std", kl.std().item(), step)
        
        # Recreate mask based on `ans_token_ids` received from actor, as it reflects the true generation length per sequence
        max_ans_len = max(len(x) for x in ans_token_ids)
        mask = torch.zeros(len(ans_token_ids), max_ans_len, device=engine.device, dtype=torch.float32)
        for i, tokens in enumerate(ans_token_ids):
            mask[i, :len(tokens)] = 1.0

        loss = ((per_tok_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)).mean()

        if rank == 0:
            writer.add_scalar("train/loss", loss.item(), step)

        engine.backward(loss)
        engine.step()
        grad_norm = engine.get_global_grad_norm()

        if rank == 0:
            print(f"Step: {step:05d} | Loss: {loss.item():.4f} | Mean Reward: {rewards.mean().item():.3f}")
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/reward", rewards.mean().item(), step)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm, step)
            # Extensive logging from grpo_vllm.py
            rewards_tensor = torch.tensor(batch['rewards'])
            completion_lengths = torch.tensor([len(x) for x in batch['ans_token_ids']], dtype=torch.float32)
            
            writer.add_scalar("gen/time", batch['gen_time'], step)
            writer.add_scalar("gen/rewards_mean", rewards_tensor.mean().item(), step)
            writer.add_scalar("gen/rewards_std", rewards_tensor.std().item(), step)
            writer.add_scalar("gen/rewards_min", rewards_tensor.min().item(), step)
            writer.add_scalar("gen/rewards_max", rewards_tensor.max().item(), step)
            writer.add_scalar("gen/completion/mean_length", completion_lengths.mean().item(), step)
            writer.add_scalar("gen/completion/min_length", completion_lengths.min().item(), step)
            writer.add_scalar("gen/completion/max_length", completion_lengths.max().item(), step)

            # Table logging
            prompts = batch['prompts']
            answers = batch['answers']
            num_prompts_in_batch = len(prompts)
            num_rollouts_per_prompt = len(answers) // num_prompts_in_batch
            
            # Group rewards by prompt to calculate advantages, mirroring the main computation
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

    # cleanup
    if rank == 0:
        req_q.put({"type": "STOP"})
        actor.join()


if __name__ == "__main__":
    main() 