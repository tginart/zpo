"""
GRPO trainer script - runs on GPUs 0-6 with FSDP
Communicates with actor.py via torch.distributed
"""

import os, random, time, json, argparse
from typing import List, Dict, Any, Optional
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
    print(f"[find_entropy_split] Called with segment of length {segment.length}, min_len={min_len}")
    if segment.length < 2 * min_len:
        print(f"[find_entropy_split] Segment too short to split: {segment.length} < 2*{min_len}={2*min_len}")
        return None
    max_entropy = -1.0
    split_idx = -1
    entropies = []
    for i in range(min_len, segment.length - min_len + 1):
        entropy = segment.actions[i - 1].entropy
        entropies.append((i, entropy))
        # print(f"[find_entropy_split] Checking split at {i}: entropy={entropy:.4f}")
        if entropy > max_entropy:
            # print(f"[find_entropy_split] New max entropy found: {entropy:.4f} at split {i}")
            max_entropy = entropy
            split_idx = i
    # if split_idx != -1:
    #     # print(f"[find_entropy_split] Best split index: {split_idx} with max entropy {max_entropy:.4f}")
    # else:
    #     # print(f"[find_entropy_split] No valid split found.")
    # print(f"[find_entropy_split] All candidate entropies: {[(i, f'{e:.4f}') for i, e in entropies]}")
    if split_idx != -1:
        parent_prefix_len = len(segment.parent.prefix_ids) if segment.parent else 0
        left_len = split_idx
        right_len = segment.length - split_idx
        new_total_prefix_len = parent_prefix_len + left_len
        print(
            f"[find_entropy_split] Selected split index {split_idx} | "
            f"left_len={left_len}, right_len={right_len}, "
            f"prev_prefix_len={parent_prefix_len}, new_total_prefix_len={new_total_prefix_len}"
        )
        return split_idx
    return None

# ----------------------------------------------------------------------------------
# Explore-only splitting helpers (ported from aspo_vllm_async_simplified)
# ----------------------------------------------------------------------------------

# Will be overwritten after CLI args are parsed
MIN_LEN_CONST = 16
THETA_A_CONST = -999999.9


def generate_rollouts_from_segment(segment: Segment, num_rollouts: int, item, model_gen, model_ref, tokenizer, reward_fn, max_gen_tokens, prompt_len_root, device):
    """
    Generate rollouts from a given segment prefix using model_gen and compute
    per-token statistics needed for ASPO.
    Returns a list of rollout dictionaries similar to the async reference.
    """
    prefix_ids = segment.prefix_ids
    inputs = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    prompt_len = len(prefix_ids)
    attention_mask = torch.ones_like(inputs)

    # Calculate remaining token budget to ensure total length doesn't exceed initial prompt + max_gen_tokens
    already_generated = prompt_len - prompt_len_root
    remaining_budget = max_gen_tokens - already_generated
    new_max_gen_tokens = max(1, remaining_budget)

    print(f"inputs: {tokenizer.batch_decode(inputs, skip_special_tokens=True)}")


    with torch.no_grad():
        outputs = model_gen.generate(
            inputs,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            max_new_tokens=new_max_gen_tokens,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=num_rollouts,
            output_scores=True,
            return_dict_in_generate=True,
        )
    full_sequences_ids = outputs.sequences  # (num_rollouts, seq_len)
    ans_token_ids = full_sequences_ids[:, prompt_len:] # (num_rollouts, gen_len)
    answers = tokenizer.batch_decode(ans_token_ids, skip_special_tokens=True)

    # Compute rewards on the full rollout completion relative to the ROOT prompt
    ans_token_ids_full_from_root = full_sequences_ids[:, prompt_len_root:]
    answers_full = tokenizer.batch_decode(ans_token_ids_full_from_root, skip_special_tokens=True)

    # print(f"prompt len: {prompt_len}")
    # print(f"prompt len root: {prompt_len_root}")
    # print(f"already generated: {already_generated}")
    # print(f"remaining budget: {remaining_budget}")
    # print(f"new max gen tokens: {new_max_gen_tokens}")

    # convert tokens to text and print
    #print(f"inputs: {tokenizer.batch_decode(inputs, skip_special_tokens=True)}")
    # just the outputs / completions from this round
    # print(f"outputs: {answers}")
    #print(f"answers_full: {answers_full}")
    #print(f"answers: {answers}")

    rewards = reward_fn(answers_full, items=[item] * len(answers_full))

    gen_logits = torch.stack(outputs.scores, dim=1)  # (num_rollouts, gen_len, vocab)
    gen_log_softmax = torch.log_softmax(gen_logits, dim=-1)
    gen_logps = torch.gather(gen_log_softmax, -1, ans_token_ids.unsqueeze(-1)).squeeze(-1)
    entropies = torch.distributions.Categorical(logits=gen_logits).entropy()

    # Reference model log-probs
    ref_attention_mask = (full_sequences_ids != tokenizer.pad_token_id).long()
    with torch.no_grad():
        ref_logits = model_ref(full_sequences_ids, attention_mask=ref_attention_mask).logits[:, prompt_len - 1 : -1, :]
    ref_log_softmax = torch.log_softmax(ref_logits, dim=-1)
    ref_logps = torch.gather(ref_log_softmax, -1, ans_token_ids.unsqueeze(-1)).squeeze(-1)

    rollouts = []
    for j in range(full_sequences_ids.shape[0]):
        gen_part = ans_token_ids[j]
        actions = [
            Action(
                token_id=int(gen_part[k].item()),
                entropy=float(entropies[j, k].item()),
                logp=float(gen_logps[j, k].item()),
                ref_logp=float(ref_logps[j, k].item()),
            )
            for k in range(gen_part.shape[0])
        ]
        rollouts.append({
            "full_seq": full_sequences_ids[j].cpu().tolist(),
            "actions": actions,
            "reward": float(rewards[j]),
            "gen_logps": gen_logps[j].cpu().tolist(),
            "ref_logps": ref_logps[j].cpu().tolist(),
            "ans_token_ids": gen_part.cpu().tolist(),
            "prefix_len": prompt_len,
            # Store only the completion relative to this segment prefix
            "answer": answers[j],
            "prompt": item["prompt"][0]["content"] if isinstance(item["prompt"], list) else item["prompt"],
        })
    return rollouts


def split_segments(root_seg: Segment, item, model_gen, model_ref, tokenizer, reward_fn, max_gen_tokens, prompt_len_root, device):
    """Greedy explore-only splitting (no exploitation), fixed 4 rounds."""
    all_rollouts = []
    for _ in range(4):
        leaves = [s for s in get_all_segments(root_seg) if s.is_leaf and s.length >= 2 * MIN_LEN_CONST]
        # print leaves
        #print(f"get all segments: {[s.length for s in get_all_segments(root_seg)]}")
        #print(f"split_segments: leaves: {[s.length for s in leaves]}")
        rewards_vals = [s.mean for s in leaves if s.rollout_count > 0]
        global_mean = sum(rewards_vals) / len(rewards_vals) if rewards_vals else 0.0
        global_std = torch.tensor(rewards_vals).std().item() if len(rewards_vals) > 1 else 1.0
        for s in leaves:
            if s.rollout_count > 0:
                s.advantage = (s.mean - global_mean) / (global_std + 1e-8)
            else:
                s.advantage = float('-inf')
        candidates = [s for s in leaves if s.advantage >= THETA_A_CONST]
        if not candidates:
            # No candidate segments: fallback to de novo rollouts from root
            #print(f"split_segments: no candidates to split, generating de novo rollouts from root")
            rollouts = generate_rollouts_from_segment(
                root_seg, num_rollouts_split, item, model_gen, model_ref, tokenizer, reward_fn, max_gen_tokens, prompt_len_root, device
            )
            for rollout in rollouts:
                new_seg = Segment(rollout["actions"], parent=root_seg)
                root_seg.children.append(new_seg)
                new_seg.propagate_reward(rollout["reward"])
                all_rollouts.append(rollout)
            continue
        seg_to_split = max(candidates, key=lambda s: s.advantage)
        split_idx = find_entropy_split(seg_to_split, MIN_LEN_CONST)
        if split_idx is None:
            assert False, (
                f"split_segments: find_entropy_split returned None for segment length {seg_to_split.length} "
                f"(min_len={MIN_LEN_CONST})"
            )
        # Detailed logging for the actual split that will be performed
        parent_prefix_len = len(seg_to_split.parent.prefix_ids) if seg_to_split.parent else 0
        left_len = split_idx
        right_len = seg_to_split.length - split_idx
        new_total_prefix_len = parent_prefix_len + left_len
        # print(
        #     f"split_segments: Performing split at index {split_idx} | "
        #     f"left_len={left_len}, right_len={right_len}, "
        #     f"prev_prefix_len={parent_prefix_len}, new_total_prefix_len={new_total_prefix_len}"
        # )
        child = seg_to_split.split(split_idx)
        # Generate from the LEFT prefix (seg_to_split) after split, not from the right-suffix child
        rollouts = generate_rollouts_from_segment(seg_to_split, num_rollouts_split, item, model_gen, model_ref, tokenizer, reward_fn, max_gen_tokens, prompt_len_root, device)
        for rollout in rollouts:
            new_seg = Segment(rollout["actions"], parent=seg_to_split)
            seg_to_split.children.append(new_seg)
            new_seg.propagate_reward(rollout["reward"])
            all_rollouts.append(rollout)
    return all_rollouts


def construct_training_batch(rollouts: list, tokenizer):
    """Pad and package rollouts into tensors/lists for returning to trainers."""
    sequences, rewards, gen_logps, ref_logps, prefix_lens, ans_token_ids, answers, prompts = [], [], [], [], [], [], [], []
    for traj in rollouts:
        sequences.append(traj["full_seq"])
        rewards.append(traj["reward"])
        gen_logps.append(traj["gen_logps"])
        ref_logps.append(traj["ref_logps"])
        prefix_lens.append(traj["prefix_len"])
        ans_token_ids.append(traj["ans_token_ids"])
        answers.append(traj["answer"])
        prompts.append(traj["prompt"])
    max_seq_len = max(len(s) for s in sequences)
    max_gen_len = max(len(l) for l in gen_logps)
    for s in sequences:
        s.extend([tokenizer.pad_token_id] * (max_seq_len - len(s)))
    for l in gen_logps:
        l.extend([0.0] * (max_gen_len - len(l)))
    for l in ref_logps:
        l.extend([0.0] * (max_gen_len - len(l)))
    return {
        "sequences": sequences,
        "rewards": rewards,
        "gen_logps": gen_logps,
        "ref_logps": ref_logps,
        "prompt_lens": prefix_lens,
        "ans_token_ids": ans_token_ids,
        "answers": answers,
        "prompts": prompts,
    }


def collect_explore_batch(model_gen, model_ref, tokenizer, reward_fn, task_items, num_rollouts_total, max_gen_tokens, device, system_prompt):
    """Sample a prompt and generate an explore-only batch via splitting."""
    item = random.choice(task_items)
    user_prompt = item["prompt"][0]["content"] if isinstance(item["prompt"], list) else item["prompt"]
    tip_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(tip_text)
    prompt_len_root = len(prompt_tokens)
    root_seg = Segment([Action(tok, 0.0, 0.0, 0.0) for tok in prompt_tokens])

    initial_rollouts = generate_rollouts_from_segment(root_seg, num_rollouts_initial, item, model_gen, model_ref, tokenizer, reward_fn, max_gen_tokens, prompt_len_root, device)
    for r in initial_rollouts:
        new_seg = Segment(r["actions"], parent=root_seg)
        root_seg.children.append(new_seg)
        new_seg.propagate_reward(r["reward"])

    rollouts = initial_rollouts + split_segments(root_seg, item, model_gen, model_ref, tokenizer, reward_fn, max_gen_tokens, prompt_len_root, device)

    # Debug: ensure we produced exactly the requested number of rollouts
    if num_rollouts_total is not None:
        assert len(rollouts) == num_rollouts_total, (
            f"collect_explore_batch: expected {num_rollouts_total} rollouts but got {len(rollouts)}. "
            f"Parameters: min_len={MIN_LEN_CONST}, num_rollouts_initial={num_rollouts_initial}, "
            f"num_rollouts_split={num_rollouts_split}. Early termination likely."
        )

    return construct_training_batch(rollouts, tokenizer)

# ----------------------------------------------------------------------------------
# CLI Arguments
# ----------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--local_rank", type=int, default=-1, help="(internal) local rank passed by launcher")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-instruct")
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--num_rollouts", type=int, default=21)
    parser.add_argument("--num_rollouts_initial", type=int, default=5)
    parser.add_argument("--num_rollouts_split", type=int, default=4)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=7)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    # ASPO specific args
    parser.add_argument("--min_len", type=int, default=16)
    parser.add_argument("--theta_adv", type=float, default=-999999.9)
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
num_rollouts_initial = args.num_rollouts_initial
num_rollouts_split = args.num_rollouts_split
num_prompts = args.num_prompts
gradient_accumulation_steps = args.gradient_accumulation_steps
max_gen_tokens = args.max_gen_tokens
beta = args.beta
clip_param = args.clip_param
lr = args.lr
grad_clip = args.grad_clip
debug = args.debug

# Update explore-only constants with CLI overrides
MIN_LEN_CONST = args.min_len
THETA_A_CONST = args.theta_adv
num_train_devices = 7

# Batch size calculations
train_batch_size = num_prompts * num_rollouts
print(f"num_prompts: {num_prompts}")
print(f"num_rollouts: {num_rollouts}")
print(f"train_batch_size: {train_batch_size}")
micro_batch_size = args.micro_batch_size

# Micro-batch validation
if train_batch_size % micro_batch_size != 0:
    raise ValueError(f"train_batch_size ({train_batch_size}) must be divisible by micro_batch_size ({micro_batch_size})")

num_micro_batches_per_train_batch = train_batch_size // micro_batch_size

# Gradient accumulation validation: must be a multiple of (train_batch_size // micro_batch_size)
if gradient_accumulation_steps % num_micro_batches_per_train_batch != 0:
    raise ValueError(
        f"gradient_accumulation_steps ({gradient_accumulation_steps}) must be a multiple of "
        f"train_batch_size // micro_batch_size ({num_micro_batches_per_train_batch}). "
        f"This ensures you update after an integer number of full train batches."
    )

# For backward compatibility, keep total_batch_size
total_batch_size = train_batch_size

# (num_rollouts - num_rollouts_initial) must be a multiple of num_rollouts_split
assert (num_rollouts - num_rollouts_initial) % num_rollouts_split == 0, "num_rollouts - num_rollouts_initial must be a multiple of num_rollouts_split"

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
            
            # Generate batch using explore-only splitting
            batch = collect_explore_batch(
                model_gen, model_ref, tokenizer, reward_fn, task_items,
                num_rollouts, max_gen_tokens, device, system_prompt
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
    if debug:   
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
        logdir = f"runs/spo_{args.task}_{args.job_id}" if args.job_id is not None else f"runs/spo_{args.task}_{time.strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(logdir) 
        # Log each CLI arg as a separate text entry
        for k, v in vars(args).items():
            writer.add_text(f"args/{k}", str(v))

    # Wait for actor to be ready
    if debug:
        print(f"[GRPO] Rank {extended_rank} waiting for actor to be ready (barrier)...")
    dist.barrier()  # Actor will join this barrier when ready
    if debug:
        print(f"[GRPO] Rank {extended_rank} actor is ready, starting training loop")

    # Training loop
    for step in trange(1, args.steps + 1, desc="Training Steps", disable=(extended_rank != 0)):
        
        # Only rank 0 gets the full state dict and sends tensors to actor
        if extended_rank == 0:
            if debug:
                print(f"[GRPO] Rank {extended_rank}: Getting full state_dict for step {step}...")
            # Use full state dict on rank 0 - FSDP will gather from all ranks
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                full_state_dict = model.state_dict()
            if debug:
                print(f"[GRPO] Rank {extended_rank}: Got full state_dict for step {step}")
            
            if debug:
                print(f"[GRPO] Rank {extended_rank}: Sending state dict to actor...")
            
            # Send control message first
            send_metadata({"type": "STATE_DICT"}, dst_rank=7, tag=step*100)
            
            # Send state dict to actor (rank 7)
            broadcast_state_dict(full_state_dict, src_rank=0, dst_rank=7, tag_base=step*100+10)
            
            if debug:
                print(f"[GRPO] Rank {extended_rank}: Sent state dict to actor")
        else:
            # Other ranks just participate in the collective gather
            if debug:
                print(f"[GRPO] Rank {extended_rank}: Participating in state_dict gather for step {step}")
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                model.state_dict()  # Participate in collective but don't use result
            if debug:
                print(f"[GRPO] Rank {extended_rank}: Participated in state_dict gather for step {step}")
        
        # Only rank 0 waits for acknowledgment (actor will send it after collecting all shards)
        if extended_rank == 0:
            if debug:   
                print(f"[GRPO] Step {step}: Waiting for ack from actor...")
            ack = recv_metadata(src_rank=7, tag=step*100+99)  # Use offset tag for ack
            if debug:
                print(f"[GRPO] Step {step}: Got ack: {ack}")
            assert ack["status"] == "OK", "Actor failed to update weights"
            
            # Request batch generation
            if debug:
                print(f"[GRPO] Step {step}: Requesting batch from actor...")
            gen_msg = {
                "type": "GENERATE", 
                "num_rollouts": num_rollouts,
                "num_prompts": num_prompts,
                "max_gen_tokens": max_gen_tokens
            }
            send_metadata(gen_msg, dst_rank=7, tag=step*100+5)
            if debug:
                print(f"[GRPO] Step {step}: Sent batch request, waiting for batch...")
            
            if debug:
                print(f"[GRPO] Step {step}: Receiving batch from actor...")
            
            # Receive shapes and metadata first
            # shapes:
            #   - sequences: [B, S] where B=num_prompts*num_rollouts, S=max sequence length
            #   - rewards: [B]
            #   - gen_logps: [B, G] where G=max generated length per sample
            #   - ref_logps: [B, G]
            batch_info = recv_metadata(src_rank=7, tag=step*100+20)
            shapes = batch_info["shapes"]
            metadata = batch_info["metadata"]
            
            # Receive tensors
            device = torch.cuda.current_device()
            # Allocate receive buffers using advertised shapes
            sequences = torch.empty(shapes["sequences"], dtype=torch.long, device=device)
            rewards = torch.empty(shapes["rewards"], dtype=torch.float32, device=device)
            gen_logps = torch.empty(shapes["gen_logps"], dtype=torch.float32, device=device)
            ref_logps = torch.empty(shapes["ref_logps"], dtype=torch.float32, device=device)
            
            dist.recv(sequences, src=7, tag=step*100+30)
            dist.recv(rewards, src=7, tag=step*100+31)
            dist.recv(gen_logps, src=7, tag=step*100+32)
            dist.recv(ref_logps, src=7, tag=step*100+33)
            
            # Reconstruct batch dict
            # Notes on structure after .tolist():
            #   - sequences: List[List[int]] of shape [B, S]
            #   - rewards: List[float] of length B
            #   - gen_logps/ref_logps: List[List[float]] of shape [B, G]
            #   - prompt_lens: List[int] length B (per-sample prompt token counts)
            #   - ans_token_ids: List[List[int]] length B (per-sample generated token ids, len varies ≤ G)
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
            
            if debug:
                print(f"[GRPO] Step {step}: Received batch from actor")
        else:
            if debug:
                print(f"[GRPO] Rank {extended_rank}: Waiting for batch broadcast in step {step}")
            batch = None

        # Broadcast batch to all training ranks using tensor communication
        device = torch.cuda.current_device()
        
        if extended_rank == 0:
            # Rank 0 converts batch to tensors and broadcasts
            # Tensor shapes:
            #   sequences: [B, S] long
            #   rewards: [B] float32
            #   gen_old/ref_logps: [B, G] float32
            sequences = torch.tensor(batch["sequences"], dtype=torch.long, device=device)
            rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
            gen_old = torch.tensor(batch["gen_logps"], dtype=torch.float32, device=device)
            ref_logps = torch.tensor(batch["ref_logps"], dtype=torch.float32, device=device)
            
            # Broadcast metadata using small object
            # shapes captured here allow receivers to pre-allocate exact tensor sizes
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
            # sequences: [B, S], rewards: [B], gen_old/ref_logps: [B, G]
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
        # rewards/gen_old/ref_logps become bfloat16 for compute; shapes remain [B] and [B, G]
        rewards = rewards.to(torch.bfloat16)
        gen_old = gen_old.to(torch.bfloat16)
        ref_logps = ref_logps.to(torch.bfloat16)
        
        prompt_lens = metadata["prompt_lens"]
        ans_token_ids = metadata["ans_token_ids"]

        # Compute advantages per prompt
        # B = N * R (N=num_prompts_in_batch, R=num_rollouts_per_prompt)
        num_prompts_in_batch = len(set(metadata["prompts"]))
        num_rollouts_per_prompt = len(rewards) // num_prompts_in_batch
        # rewards_per_prompt: [N, R]
        rewards_per_prompt = rewards.view(num_prompts_in_batch, num_rollouts_per_prompt)
        # mean_per_prompt/std_per_prompt: [N, 1]; advantages_per_prompt: [N, R]
        mean_per_prompt = rewards_per_prompt.mean(dim=1, keepdim=True)
        std_per_prompt = rewards_per_prompt.std(dim=1, keepdim=True)
        advantages_per_prompt = (rewards_per_prompt - mean_per_prompt) / (std_per_prompt + 1e-4)
        # Use overall batch statistics for policy gradient
        # advantages: [B]
        adv_mean = rewards.mean()
        adv_std = rewards.std()
        advantages = (rewards - adv_mean) / (adv_std + 1e-4)
        if extended_rank == 0:
            # print various compute advantages quantities
            # print(f"metadata: {metadata}")
            # print(f"num_prompts_in_batch: {num_prompts_in_batch}")
            # print(f"num_rollouts_per_prompt: {num_rollouts_per_prompt}")
            # print(f"rewards_per_prompt: {rewards_per_prompt}")
            # print(f"mean_per_prompt: {mean_per_prompt}")
            # print(f"std_per_prompt: {std_per_prompt}")
            # print(f"advantages_per_prompt: {advantages_per_prompt}")
            # print(f"adv_mean: {adv_mean}")
            pass

        # Micro-batch training loop
        num_micro_batches = total_batch_size // args.micro_batch_size
        microbatch_losses = []
        microbatch_pg_losses = []
        microbatch_kl = []
        microbatch_ratios = []
        for mb_idx in range(num_micro_batches):
            start = mb_idx * args.micro_batch_size
            end = start + args.micro_batch_size
            # Per-micro-batch views
            #   mb_sequences: [mb, S]
            #   mb_rewards: [mb]
            #   mb_gen_old/mb_ref_logps: [mb, G]
            #   mb_prompt_lens: List[int] length mb
            #   mb_ans_token_ids: List[List[int]] length mb
            #   mb_advantages: [mb]
            mb_sequences = sequences[start:end]
            mb_rewards = rewards[start:end]
            mb_gen_old = gen_old[start:end]
            mb_ref_logps = ref_logps[start:end]
            mb_prompt_lens = prompt_lens[start:end]
            mb_ans_token_ids = ans_token_ids[start:end]
            mb_advantages = advantages[start:end]

            # Forward pass
            # mb_logits: [mb, S, V]; mb_full_logps: [mb, S-1]
            mb_logits = model(mb_sequences).logits
            mb_full_logps = per_token_logps(mb_logits[:, :-1, :], mb_sequences[:, 1:])
            gen_len = mb_gen_old.shape[1]  # target width G
            curr_logps_list = []
            for i in range(mb_full_logps.shape[0]):
                p_len = mb_prompt_lens[i]
                ans_len = len(mb_ans_token_ids[i])
                # sliced_logps: [ans_len]; padded to [G]
                sliced_logps = mb_full_logps[i, p_len - 1 : p_len - 1 + ans_len]
                padded_logps = F.pad(sliced_logps, (0, gen_len - ans_len), 'constant', 0)
                curr_logps_list.append(padded_logps)
            mb_curr_logps = torch.stack(curr_logps_list, dim=0)  # [mb, G]

            # PPO loss computation with numerical stability
            # Shapes below are all [mb, G] unless noted
            ratio_diff = torch.clamp(mb_curr_logps - mb_gen_old.detach(), min=-20.0, max=20.0)
            kl_diff = torch.clamp(mb_ref_logps - mb_curr_logps, min=-20.0, max=20.0)
            mb_ratios = torch.exp(ratio_diff)
            mb_clipped = torch.clamp(mb_ratios, 1 - clip_param, 1 + clip_param)
            mb_advantages = mb_advantages.unsqueeze(-1)  # [mb, 1]
            mb_pg_loss = -torch.min(mb_ratios * mb_advantages, mb_clipped * mb_advantages)
            mb_kl = torch.exp(kl_diff) - kl_diff - 1
            mb_per_tok_loss = mb_pg_loss + beta * mb_kl

            # Create mask for loss computation
            # Use gen_len to match the width of mb_per_tok_loss
            # mask: [mb, G] with 1.0 over valid tokens and 0.0 over padding
            mask = torch.zeros(len(mb_ans_token_ids), gen_len, device=device, dtype=torch.float32)
            for i, tokens in enumerate(mb_ans_token_ids):
                mask[i, :len(tokens)] = 1.0

            # Per-sample reduction over tokens → [mb], then mean over micro-batch → scalar
            mb_loss = ((mb_per_tok_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)).mean()
            mb_loss = mb_loss / gradient_accumulation_steps
            mb_loss.backward()

            # Accumulate for logging (flatten per-token tensors)
            microbatch_losses.append(mb_loss.detach().cpu())
            microbatch_pg_losses.append(mb_pg_loss.detach().cpu().flatten())
            microbatch_kl.append(mb_kl.detach().cpu().flatten())
            microbatch_ratios.append(mb_ratios.detach().cpu().flatten())

        # Gradient accumulation: only step optimizer every gradient_accumulation_steps
        if step % gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        else:
            grad_norm = None

        # Logging (rank 0 only)
        if extended_rank == 0:
            # Use micro-batch statistics for logging, matching grpo.py
            loss = torch.stack(microbatch_losses).mean().item()
            pg_loss_all = torch.cat(microbatch_pg_losses)
            kl_all = torch.cat(microbatch_kl)
            ratios_all = torch.cat(microbatch_ratios)
            print(f"Step: {step:05d} | Loss: {loss:.4f} | Mean Reward: {rewards.mean().item():.3f}")
            writer.add_scalar("train/loss", loss, step)
            writer.add_scalar("train/reward", rewards.mean().item(), step)
            writer.add_scalar("train/gen_time", batch['gen_time'], step)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm, step)
            # Rewards per prompt (reuse tensors and grouping computed above)
            rewards_tensor = rewards.detach().float().cpu()
            prompts = metadata['prompts']
            answers = batch['answers']
            # Reuse precomputed group sizes and per-prompt stats; move small stats to CPU for logging
            mean_per_prompt_cpu = mean_per_prompt.detach().float().cpu()
            std_per_prompt_cpu = std_per_prompt.detach().float().cpu()
            writer.add_scalar("train/rewards_per_prompt/mean", mean_per_prompt_cpu.mean().item(), step)
            writer.add_scalar("train/rewards_per_prompt/std", std_per_prompt_cpu.mean().item(), step)
            writer.add_scalar("train/ratios/mean", ratios_all.mean().item(), step)
            writer.add_scalar("train/ratios/std", ratios_all.std().item(), step)
            writer.add_scalar("train/kl/mean", kl_all.mean().item(), step)
            writer.add_scalar("train/kl/std", kl_all.std().item(), step)
            writer.add_scalar("train/pg_loss/mean", pg_loss_all.mean().item(), step)
            writer.add_scalar("train/pg_loss/std", pg_loss_all.std().item(), step)
            # Advantages per prompt (use CPU copies; guard against NaNs/zeros)
            rewards_view_cpu = rewards_tensor.view(num_prompts_in_batch, num_rollouts_per_prompt)
            std_safe_cpu = torch.where(torch.isnan(std_per_prompt_cpu) | (std_per_prompt_cpu == 0), torch.full_like(std_per_prompt_cpu, 1e-4), std_per_prompt_cpu)
            advantages_per_prompt_cpu = (rewards_view_cpu - mean_per_prompt_cpu) / (std_safe_cpu + 1e-4)
            writer.add_scalar("train/advantages_per_prompt/mean", advantages_per_prompt_cpu.mean().item(), step)
            writer.add_scalar("train/advantages_per_prompt/std", advantages_per_prompt_cpu.std().item(), step)
            # Completion length stats
            completion_lengths = torch.tensor([len(x) for x in ans_token_ids], dtype=torch.float32)
            writer.add_scalar("gen/completion/mean_length", completion_lengths.mean().item(), step)
            writer.add_scalar("gen/completion/min_length", completion_lengths.min().item(), step)
            writer.add_scalar("gen/completion/max_length", completion_lengths.max().item(), step)
            # Reward stats
            writer.add_scalar("gen/rewards_mean", rewards_tensor.mean().item(), step)
            writer.add_scalar("gen/rewards_std", rewards_tensor.std().item(), step)
            writer.add_scalar("gen/rewards_min", rewards_tensor.min().item(), step)
            writer.add_scalar("gen/rewards_max", rewards_tensor.max().item(), step)
            # Table logging (align with grpo.py; reuse computed stats)
            sequences_list = batch['sequences']
            prompt_lens = batch['prompt_lens']
            advantages_for_logging = advantages_per_prompt_cpu.flatten()
            for p_idx in range(num_prompts_in_batch):
                prompt_text = prompts[p_idx]
                S = f"Prompt: {prompt_text}<br><br>"
                for r_idx in range(num_rollouts_per_prompt):
                    ans_idx = p_idx * num_rollouts_per_prompt + r_idx
                    prefix_ids = sequences_list[ans_idx][:prompt_lens[ans_idx]]
                    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
                    completion_text = answers[ans_idx]
                    reward_val = rewards_tensor[ans_idx].item()
                    adv_val = advantages_for_logging[ans_idx].item()
                    S += f"Prefix: {prefix_text}<br><br>"
                    S += f"Completion: {completion_text}<br><br>"
                    S += f"Reward: {reward_val}<br><br>"
                    S += f"Advantage: {adv_val}<br><br>"
                writer.add_text("generations", S, step)

    print(f"[GRPO] Rank {extended_rank} training completed")

    # Cleanup - send stop signal to actor (only rank 0)
    if extended_rank == 0:
        if debug:
            print(f"[GRPO] Rank {extended_rank}: Sending stop signal to actor...")
        send_metadata({"type": "STOP"}, dst_rank=7, tag=(args.steps+1)*100)
        if debug:
            print(f"[GRPO] Rank {extended_rank}: Sent stop signal to actor")
    
    # Proper cleanup
    print(f"[GRPO] Rank {extended_rank} cleaning up...")
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"[GRPO] Rank {extended_rank} cleanup warning: {e}")

if __name__ == "__main__":
    main()