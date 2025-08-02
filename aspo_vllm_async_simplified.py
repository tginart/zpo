"""
Simplified asynchronous ASPO script (exploration-only)
====================================================
This version keeps the same distributed learner/actor architecture as
`aspo_vllm_sync_ds_multi.py` but **removes all splitting / exploitation logic**.
Each training step proceeds as follows:

1. Rank-0 (the *conductor*) broadcasts current policy weights to the actor.
2. The actor generates **28 rollouts** (fixed) for **one prompt** sampled from
the task dataset.
3. The conductor packages the trajectories into a batch and broadcasts it to
   all learner ranks.
4. All ranks perform the PPO/GRPO-style loss and update the policy with
   DeepSpeed.

All DeepSpeed hyper-parameters and the training-phase loss calculation are
identical to `grpo_vllm_sync_ds_multi.py`.  The only differences are:

*   Always exactly **1 prompt** and **28 rollouts** per step.
*   No ASPO segmentation / splitting.
*   Micro batch size is fixed to **4** per GPU (4 × 7 GPUs = 28 sequences).

This stripped-down version should be much easier to debug.
"""

import os, random, time, argparse, json
from typing import List, Dict, Any, Optional

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
# ASPO Segment Tree Classes and Helpers (faithful copy from sync_ds_multi)
# ----------------------------------------------------------------------------------

class Action:
    """Represents one token sampled during generation, with its policy entropy."""
    def __init__(self, token_id: int, entropy: float, logp: float, ref_logp: float):
        self.token_id = token_id
        self.entropy = entropy
        self.logp = logp
        self.ref_logp = ref_logp

class Segment:
    """A sequence of actions forming a node in the ASPO search tree."""
    def __init__(self, actions: list, parent: Optional['Segment'] = None):
        self.actions = actions
        self.parent = parent
        self.children: list = []
        # Running reward statistics (updated via Welford's algorithm)
        self.rollout_count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squares of differences from the mean
        # ASPO metrics calculated at the start of each generation loop
        self.advantage = 0.0
        self.branch_se = 0.0
    @property
    def variance(self) -> float:
        if self.rollout_count < 2:
            return 0.0
        return self.m2 / (self.rollout_count - 1)
    @property
    def length(self) -> int:
        return len(self.actions)
    @property
    def is_leaf(self) -> bool:
        return not self.children
    @property
    def prefix_ids(self) -> list:
        if self.parent:
            return self.parent.prefix_ids + [a.token_id for a in self.actions]
        return [a.token_id for a in self.actions]
    @property
    def full_logps(self) -> list:
        if self.parent:
            return self.parent.full_logps + [a.logp for a in self.actions]
        return [a.logp for a in self.actions]
    @property
    def full_ref_logps(self) -> list:
        if self.parent:
            return self.parent.full_ref_logps + [a.ref_logp for a in self.actions]
        return [a.ref_logp for a in self.actions]
    def split(self, split_idx: int) -> 'Segment':
        if not (0 < split_idx < self.length):
            raise ValueError("Split index must be internal to the segment.")
        child_actions = self.actions[split_idx:]
        self.actions = self.actions[:split_idx]
        child = Segment(child_actions, parent=self)
        # Child inherits parent's stats before the split
        child.rollout_count = self.rollout_count
        child.mean = self.mean
        child.m2 = self.m2
        # The new child becomes the first child. Existing children of the parent
        # are re-parented to this new child. This maintains the tree structure
        # where children represent alternative suffixes.
        child.children = self.children
        for c in child.children:
            c.parent = child
        self.children = [child]
        return child
    def propagate_reward(self, reward: float):
        node = self
        while node is not None:
            node.rollout_count += 1
            delta = reward - node.mean
            node.mean += delta / node.rollout_count
            delta2 = reward - node.mean
            node.m2 += delta * delta2
            node = node.parent
    def __repr__(self):
        return f"Segment(len={self.length}, count={self.rollout_count}, mean={self.mean:.2f}, advantage={self.advantage:.2f})"

def get_all_segments(root: Segment) -> list:
    """Traverse the tree and return a flat list of all segments."""
    segments = []
    nodes_to_visit = [root]
    while nodes_to_visit:
        node = nodes_to_visit.pop(0)
        segments.append(node)
        nodes_to_visit.extend(node.children)
    return segments

def find_entropy_split(segment: Segment, min_len: int) -> Optional[int]:
    """
    Finds the best split point in a segment based on raw policy entropy.
    The split must respect the `min_len` constraint for both the resulting
    parent and child segments.
    """
    if segment.length < 2 * min_len:
        return None
    max_entropy = -1.0
    split_idx = -1
    for i in range(min_len, segment.length - min_len + 1):
        entropy = segment.actions[i - 1].entropy
        if entropy > max_entropy:
            max_entropy = entropy
            split_idx = i
    return split_idx if split_idx != -1 else None

# -------------------------------------------------------------------------------------------------
# CLI ARGUMENTS
# -------------------------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_gpu", type=int, default=1, help="Global GPU id reserved for the generation actor")
    parser.add_argument("--local_rank", type=int, default=-1, help="(internal) local rank passed by DeepSpeed")
    parser.add_argument("--train_gpus", type=int, default=7, help="Number of training devices (DeepSpeed data parallel)")

    # Core training params
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-instruct")
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_gen_tokens", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-6)

    # PPO loss params
    parser.add_argument("--beta", type=float, default=0.001, help="KL penalty coefficient")
    parser.add_argument("--clip_param", type=float, default=0.1)

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

args = get_args()

# -------------------------------------------------------------------------------------------------
# GLOBAL CONSTANTS & HYPER-PARAMETERS
# -------------------------------------------------------------------------------------------------

task_name: str = args.task
model_path: str = args.model_path
num_rollouts: int = 21            # <-- fixed rollouts per prompt
num_prompts: int = 1              # always one prompt per step
max_gen_tokens: int = args.max_gen_tokens
beta: float = args.beta
clip_param: float = args.clip_param
lr: float = args.lr
num_train_devices: int = args.train_gpus  # expected 7

# Exploitation-only ASPO segment split parameters
MIN_LEN = 16  # Minimum segment length for splitting (must have at least 2*MIN_LEN tokens)
THETA_A = -0.1  # Always split if possible (greedy exploitation)

grad_accum: int = args.gradient_accumulation_steps

# DeepSpeed micro batch size (per GPU) is fixed at 4 so that 4 × 7 = 28 sequences per step.
TRAIN_MICRO_BATCH_SIZE = 1
assert num_rollouts % (TRAIN_MICRO_BATCH_SIZE * num_train_devices) == 0, "num_rollouts must be a multiple of (TRAIN_MICRO_BATCH_SIZE * num_train_devices)"
# make sure gradient accumulation is a multiple of  num_rollouts / (TRAIN_MICRO_BATCH_SIZE * num_train_devices)
assert grad_accum % (num_rollouts / (TRAIN_MICRO_BATCH_SIZE * num_train_devices)) == 0, "gradient accumulation must be a multiple of num_rollouts / (TRAIN_MICRO_BATCH_SIZE * num_train_devices)"

# DeepSpeed runtime config
DS_CONFIG: Dict[str, Any] = {
    "train_micro_batch_size_per_gpu": TRAIN_MICRO_BATCH_SIZE,
    "gradient_accumulation_steps": grad_accum,
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
    "gradient_clipping": 1.0,
}

# -------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------------------------------

def per_token_logps(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Return log-probs of *ids* under *logits* (same as in GRPO)."""
    logp = torch.log_softmax(logits, dim=-1)
    return torch.gather(logp, -1, ids.unsqueeze(-1)).squeeze(-1)

# Simple helper for numeric stats

def _tensor_stats(t: torch.Tensor):
    return float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).min()), float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).max())


# -------------------------------------------------------------------------------------------------
# ACTOR PROCESS (runs on its own dedicated GPU)
# -------------------------------------------------------------------------------------------------

def actor_process(actor_gpu: int, req_q: mp.Queue, resp_q: mp.Queue, max_gen_tokens: int):
    # Ensure this process only sees the actor GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(actor_gpu)
    import torch  # re-import under the new CUDA context

    actor_device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Separate policy and reference models (reference stays frozen)
    model_gen = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(actor_device)
    model_ref = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(actor_device)
    model_gen.eval(); model_ref.eval()

    # Signal readiness to the learner
    resp_q.put({"status": "READY"})

    # task utilities (reward function)
    task_items, reward_fn = get_task(task_name)

    system_prompt = "You are a helpful assistant."

    while True:
        msg = req_q.get()
        mtype = msg["type"]

        if mtype == "STOP":
            break
        elif mtype == "UPDATE_WEIGHTS":
            model_gen.load_state_dict(msg["state_dict"]); resp_q.put({"status": "OK"})
            if False:
                bad = []
                for n, p in model_gen.named_parameters():
                    if not torch.isfinite(p).all():
                        bad.append(n)
                if bad:
                    print(f"[Actor][WARN] Non-finite parameters detected after load: {bad[:10]} ... total {len(bad)}")
                else:
                    # print stats of one parameter
                    n, p = next(iter(model_gen.named_parameters()))
                    print(f"[Actor] Param stats {n}: {_tensor_stats(p)}")
                if False:
                    print(f"[Actor] Model param stats after load:")
                    for n, p in model_gen.named_parameters():
                        print(f"    {n}: min={p.min().item():.4e}, max={p.max().item():.4e}, mean={p.mean().item():.4e}, std={p.std().item():.4e}")
        elif mtype == "GENERATE":
            tic = time.time()
            # Support arbitrary prefix_ids and num_rollouts for exploitation-only ASPO
            prefix_ids = msg["prefix_ids"]
            num_rollouts = msg["num_rollouts"]
            reward_context = msg["item"]
            # Prepare input for generation
            inputs = torch.tensor([prefix_ids], device=actor_device)
            prompt_len = len(prefix_ids)
            # Generate rollouts
            outputs = model_gen.generate(
                inputs,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                max_new_tokens=max_gen_tokens,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=num_rollouts,
                output_scores=True,
                return_dict_in_generate=True,
            )
            full_sequences_ids = outputs.sequences
            ans_token_ids = full_sequences_ids[:, prompt_len:]
            answers = tokenizer.batch_decode(ans_token_ids, skip_special_tokens=True)
            rewards = reward_fn(answers, items=[reward_context] * len(answers))
            # Log-probs and entropies for generated tokens
            gen_logits_gen_model = torch.stack(outputs.scores, dim=1)
            gen_log_softmax = torch.log_softmax(gen_logits_gen_model, dim=-1)
            gen_logps = torch.gather(gen_log_softmax, -1, ans_token_ids.unsqueeze(-1)).squeeze(-1)
            softmax = torch.softmax(gen_logits_gen_model, dim=-1)
            log_softmax = torch.log_softmax(gen_logits_gen_model, dim=-1)
            entropies = -(softmax * log_softmax).sum(dim=-1)
            # Reference model logps
            with torch.no_grad():
                ref_logits = model_ref(full_sequences_ids).logits[:, prompt_len - 1 : -1, :]
            ref_log_softmax = torch.log_softmax(ref_logits, dim=-1)
            ref_logps = torch.gather(ref_log_softmax, -1, ans_token_ids.unsqueeze(-1)).squeeze(-1)
            resp_q.put({
                "full_sequences": full_sequences_ids.cpu().tolist(),
                "rewards": rewards,
                "gen_logps": gen_logps.cpu().tolist(),
                "ref_logps": ref_logps.cpu().tolist(),
                "entropies": entropies.cpu().tolist(),
                "prompt_len": prompt_len,
                "ans_token_ids": ans_token_ids.cpu().tolist(),
                "answers": answers,
            })
        else:
            raise RuntimeError(f"Unknown msg type {mtype}")

# -------------------------------------------------------------------------------------------------
# MAIN LEARNER FUNCTION (executed by every rank)
# -------------------------------------------------------------------------------------------------

def main():
    deepspeed.init_distributed()
    rank = dist.get_rank(); world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"[Rank {rank}/{world_size}] Initialized. Local rank: {local_rank}")

    # ------------------------------------------------------------------
    # Rank-0 launches the actor process
    # ------------------------------------------------------------------
    if rank == 0:
        mp.set_start_method("spawn", force=True)
        req_q, resp_q = mp.Queue(), mp.Queue()
        actor = mp.Process(target=actor_process, args=(args.actor_gpu, req_q, resp_q, max_gen_tokens))
        actor.start()
    else:
        req_q = resp_q = None  # type: ignore

    # Load tokenizer & model (all ranks)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    engine, optimizer, _, _ = deepspeed.initialize(config=DS_CONFIG, model=model, model_parameters=model.parameters())
    print(f"[Rank {rank}] DeepSpeed engine ready.")

    if rank == 0:
        writer = SummaryWriter(f"runs/aspo_async_simplified_{time.strftime('%Y%m%d-%H%M%S')}")
        writer.add_text("args", json.dumps(vars(args)))

    # Wait for actor to finish loading
    if rank == 0:
        if resp_q.get().get("status") != "READY":
            raise RuntimeError("Actor failed to initialize")
    dist.barrier()

    # Preload task items on rank-0
    if rank == 0:
        task_items, _ = get_task(task_name)
    else:
        task_items = None  # type: ignore

    # --------------------------------------------------------------------------------------------------
    # TRAINING LOOP (computational steps = micro-batch backward passes)
    # --------------------------------------------------------------------------------------------------
    logical_batch_size = num_rollouts  # e.g., 14
    micro_batch_size = TRAIN_MICRO_BATCH_SIZE * num_train_devices  # e.g., 7
    num_micro_batches = logical_batch_size // micro_batch_size
    assert logical_batch_size % micro_batch_size == 0, "logical batch size must be divisible by micro batch size"

    total_computational_steps = args.steps
    step = 0
    param_update = 0
    mb_data = []
    while step < total_computational_steps:
        # Generate a new logical batch every num_micro_batches steps
        if step % num_micro_batches == 0:
            if rank == 0:
                state_dict_cpu = {k: v.to("cpu") for k, v in engine.module.state_dict().items()}
                req_q.put({"type": "UPDATE_WEIGHTS", "state_dict": state_dict_cpu}); resp_q.get()
                item = random.choice(task_items)  # single prompt
                # Build chat prompt and tokenize
                system_prompt = "You are a helpful assistant."
                user_prompt = item["prompt"][0]["content"] if isinstance(item["prompt"], list) else item["prompt"]
                tip_text = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
                prompt_tokens = tokenizer.encode(tip_text)
                # Create root segment with dummy actions for prompt tokens
                root_seg = Segment([Action(tok, 0.0, 0.0, 0.0) for tok in prompt_tokens])
                all_rollouts = []
                # --- 1. Initial 5 rollouts from root ---
                req_q.put({
                    "type": "GENERATE",
                    "prefix_ids": root_seg.prefix_ids,
                    "num_rollouts": 5,
                    "item": item,
                })
                res = resp_q.get()
                for j in range(len(res["full_sequences"])):
                    full_seq = res["full_sequences"][j]
                    gen_part = full_seq[root_seg.length:]
                    actions = [Action(gen_part[k], res["entropies"][j][k], res["gen_logps"][j][k], res["ref_logps"][j][k]) for k in range(len(gen_part))]
                    new_segment = Segment(actions, parent=root_seg)
                    root_seg.children.append(new_segment)
                    new_segment.propagate_reward(res["rewards"][j])
                    all_rollouts.append({
                        "prefix_len": root_seg.length,
                        "sequence": full_seq,
                        "reward": res["rewards"][j],
                        "gen_logps": [a.logp for a in actions],
                        "ref_logps": [a.ref_logp for a in actions],
                        "ans_token_ids": [t for t in full_seq[root_seg.length:]],
                    })
                # --- 2. 4 rounds of exploitation-only splitting ---
                for round_idx in range(4):
                    # Find all leaf segments
                    all_segments = get_all_segments(root_seg)
                    leaves = [s for s in all_segments if s.is_leaf and s.length >= 2 * MIN_LEN]
                    # Compute global mean/std and advantages
                    rewards = [s.mean for s in leaves if s.rollout_count > 0]
                    global_mean = sum(rewards) / len(rewards) if rewards else 0.0
                    global_std = torch.tensor(rewards).std().item() if len(rewards) > 1 else 1.0
                    for s in leaves:
                        if s.rollout_count > 0:
                            s.advantage = (s.mean - global_mean) / (global_std + 1e-8)
                    # Select best segment to split
                    candidates = [s for s in leaves if s.advantage >= THETA_A]
                    if not candidates:
                        break
                    seg_to_split = max(candidates, key=lambda s: s.advantage)
                    split_idx = find_entropy_split(seg_to_split, MIN_LEN)
                    if split_idx is None:
                        break
                    child = seg_to_split.split(split_idx)
                    # Generate 4 rollouts from the new segment
                    req_q.put({
                        "type": "GENERATE",
                        "prefix_ids": child.prefix_ids,
                        "num_rollouts": 4,
                        "item": item,
                    })
                    res = resp_q.get()
                    for j in range(len(res["full_sequences"])):
                        full_seq = res["full_sequences"][j]
                        gen_part = full_seq[child.length:]
                        actions = [Action(gen_part[k], res["entropies"][j][k], res["gen_logps"][j][k], res["ref_logps"][j][k]) for k in range(len(gen_part))]
                        new_segment = Segment(actions, parent=child)
                        child.children.append(new_segment)
                        new_segment.propagate_reward(res["rewards"][j])
                        all_rollouts.append({
                            "prefix_len": child.length,
                            "sequence": full_seq,
                            "reward": res["rewards"][j],
                            "gen_logps": [a.logp for a in actions],
                            "ref_logps": [a.ref_logp for a in actions],
                            "ans_token_ids": [t for t in full_seq[child.length:]],
                        })
                # --- 3. Batch construction ---
                sequences, rewards, gen_logps, ref_logps, prompt_lens, ans_token_ids = [], [], [], [], [], []
                for traj in all_rollouts:
                    sequences.append(traj["sequence"])
                    rewards.append(traj["reward"])
                    prompt_lens.append(traj["prefix_len"])
                    ans_token_ids.append(traj["ans_token_ids"])
                    gen_logps.append(traj["gen_logps"])
                    ref_logps.append(traj["ref_logps"])
                # Pad for batching
                max_seq_len = max(len(s) for s in sequences)
                max_gen_len = max(len(l) for l in gen_logps)
                for s in sequences: s.extend([tokenizer.pad_token_id] * (max_seq_len - len(s)))
                for l in gen_logps: l.extend([0.0] * (max_gen_len - len(l)))
                for l in ref_logps: l.extend([0.0] * (max_gen_len - len(l)))
                batch_raw = {
                    "sequences": sequences, "rewards": rewards, "gen_logps": gen_logps,
                    "ref_logps": ref_logps, "prompt_lens": prompt_lens,
                    "ans_token_ids": ans_token_ids,
                }
            else:
                batch_raw = None
            # Broadcast batch to all ranks
            obj_list = [batch_raw]; dist.broadcast_object_list(obj_list, src=0)
            if rank != 0:
                batch_raw = obj_list[0]
            # Package tensors for the full logical batch
            sequences = torch.tensor(batch_raw["sequences"], dtype=torch.long, device=engine.device)
            rewards = torch.tensor(batch_raw["rewards"], dtype=torch.bfloat16, device=engine.device)
            gen_old = torch.tensor(batch_raw["gen_logps"], dtype=torch.bfloat16, device=engine.device)
            ref_logps = torch.tensor(batch_raw["ref_logps"], dtype=torch.bfloat16, device=engine.device)
            prompt_lens = batch_raw["prompt_lens"]
            ans_token_ids = batch_raw["ans_token_ids"]
            # Compute advantages over the full batch
            adv_mean, adv_std = rewards.mean(), rewards.std()
            advantages = (rewards - adv_mean) / (adv_std + 1e-4)
            if rank == 0:
                writer.add_scalar("train/reward_mean", rewards.mean().item(), step)
                writer.add_scalar("train/reward_std", rewards.std().item(), step)
            # Split the full batch into micro-batches for accumulation
            mb_data = []
            for mb_idx in range(num_micro_batches):
                mb_start = mb_idx * micro_batch_size
                mb_end = (mb_idx + 1) * micro_batch_size
                mb_data.append((
                    sequences[mb_start:mb_end],
                    rewards[mb_start:mb_end],
                    gen_old[mb_start:mb_end],
                    ref_logps[mb_start:mb_end],
                    advantages[mb_start:mb_end],
                    ans_token_ids[mb_start:mb_end],
                ))
        # Now process the next micro-batch
        mb_idx = step % num_micro_batches
        mb_sequences, mb_rewards, mb_gen_old, mb_ref_logps, mb_advantages, mb_ans_token_ids = mb_data[mb_idx]
        # Forward pass
        logits = engine(mb_sequences).logits
        full_logps = per_token_logps(logits[:, :-1, :], mb_sequences[:, 1:])
        # Slice log-probs of generated tokens (variable length handling)
        gen_len = mb_gen_old.shape[1]
        curr_logps_list = []
        for i in range(full_logps.shape[0]):
            p_len = prompt_lens[i]
            ans_len = len(mb_ans_token_ids[i])
            sliced = full_logps[i, p_len - 1 : p_len - 1 + ans_len]
            padded = F.pad(sliced, (0, gen_len - ans_len), value=0.0)
            curr_logps_list.append(padded)
        curr_logps = torch.stack(curr_logps_list, dim=0)
        # PPO-style loss (same as GRPO)
        ratios = torch.exp(curr_logps - mb_gen_old.detach())
        clipped = torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
        pg_loss = -torch.min(ratios * mb_advantages.unsqueeze(-1), clipped * mb_advantages.unsqueeze(-1))
        kl = torch.exp(mb_ref_logps - curr_logps) - (mb_ref_logps - curr_logps) - 1
        per_tok_loss = pg_loss + beta * kl
        # Mask (based on ans_token_ids) to ignore padding
        max_ans_len = max(len(x) for x in mb_ans_token_ids)
        mask = torch.zeros(len(mb_ans_token_ids), max_ans_len, device=engine.device, dtype=torch.float32)
        for i, toks in enumerate(mb_ans_token_ids):
            mask[i, : len(toks)] = 1.0
        loss = ((per_tok_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)).mean()
        # Backprop / optimizer step (accumulate gradients)
        if not torch.isfinite(loss):
            if rank == 0:
                print("[Learner][FATAL] Loss became non-finite!", loss.item())
            break
        engine.backward(loss)
        # Only update parameters every grad_accum steps
        if ((step + 1) % grad_accum) == 0:
            engine.step()
            grad_norm = engine.get_global_grad_norm()
            param_update += 1
            if rank == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                print(f"Step {step+1:05d} | ParamUpdate {param_update:05d} | Loss {loss.item():.4f} | Reward {rewards.mean().item():.3f}")
        # Optionally, add debug numerics here as before
        step += 1

    # Clean up
    if rank == 0:
        req_q.put({"type": "STOP"}); actor.join()


if __name__ == "__main__":
    main() 