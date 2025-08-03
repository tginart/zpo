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
    return split_idx if split_idx != -1 else None

# -------------------------------------------------------------------------------------------------
# CLI ARGUMENTS
# -------------------------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_gpu", type=int, default=7, help="Global GPU id reserved for the generation actor")
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
THETA_A = -9999999999  # Always split if possible (greedy exploitation)

grad_accum: int = args.gradient_accumulation_steps

# DeepSpeed micro batch size (per GPU) is fixed at 4 so that 4 × 7 = 28 sequences per step.
TRAIN_MICRO_BATCH_SIZE = 1
assert num_rollouts % (TRAIN_MICRO_BATCH_SIZE * num_train_devices) == 0, "num_rollouts must be a multiple of (TRAIN_MICRO_BATCH_SIZE * num_train_devices)"
# make sure gradient accumulation is a multiple of  num_rollouts / (TRAIN_MICRO_BATCH_SIZE * num_train_devices)
logical_micro_batches = num_rollouts // (TRAIN_MICRO_BATCH_SIZE * num_train_devices)
assert grad_accum % logical_micro_batches == 0, "gradient accumulation must be a multiple of num_rollouts / (TRAIN_MICRO_BATCH_SIZE * num_train_devices)"

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
    # All 8 GPUs are now visible to this process (0-7)
    # The actor will use GPU 7 directly
    import torch  # re-import under the new CUDA context

    # Use GPU 7 for the actor
    actor_device = torch.device("cuda:7")

    # Print some debugging information
    print(f"[ACTOR DEBUG] Using GPU 7 for actor")
    print(f"[ACTOR DEBUG] Visible devices: {torch.cuda.device_count()}")
    print(f"[ACTOR DEBUG] Device name (cuda:7): {torch.cuda.get_device_name(7)}")

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
            # print shape of inputs
            print(f"[ACTOR DEBUG] inputs shape: {inputs.shape}")
            # Build attention mask for the input (all 1s for prompt)
            attention_mask = torch.ones_like(inputs)
            # Generate rollouts
            outputs = model_gen.generate(
                inputs,
                attention_mask=attention_mask,
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
            print(f"[ACTOR DEBUG] full_sequences_ids shape: {full_sequences_ids.shape}")
            ans_token_ids = full_sequences_ids[:, prompt_len:]
            answers = tokenizer.batch_decode(ans_token_ids, skip_special_tokens=True)
            rewards = reward_fn(answers, items=[reward_context] * len(answers))
            # Log-probs and entropies for generated tokens
            gen_logits_gen_model = torch.stack(outputs.scores, dim=1)
            # print(f"[ACTOR DEBUG] gen_logits_gen_model shape: {gen_logits_gen_model.shape}")
            # print(f"[ACTOR DEBUG] gen_logits_gen_model min/max: {gen_logits_gen_model.min().item()}, {gen_logits_gen_model.max().item()}")
            # print(f"[ACTOR DEBUG] gen_logits_gen_model has_nan: {torch.isnan(gen_logits_gen_model).any().item()}")
            # print(f"[ACTOR DEBUG] gen_logits_gen_model has_inf: {torch.isinf(gen_logits_gen_model).any().item()}")
            gen_log_softmax = torch.log_softmax(gen_logits_gen_model, dim=-1)
            print(f"[ACTOR DEBUG] gen_log_softmax shape: {gen_log_softmax.shape}")
            print(f"[ACTOR DEBUG] gen_log_softmax min/max: {gen_log_softmax.min().item()}, {gen_log_softmax.max().item()}")
            gen_logps = torch.gather(gen_log_softmax, -1, ans_token_ids.unsqueeze(-1)).squeeze(-1)
            print(f"[ACTOR DEBUG] gen_logps shape: {gen_logps.shape}")
            print(f"[ACTOR DEBUG] gen_logps min/max: {gen_logps.min().item()}, {gen_logps.max().item()}")

            # === NEW DEBUG PRINTS FOR TOKEN ALIGNMENT ===
            # print("[ACTOR DEBUG] gen_logps min/max:", gen_logps.min().item(), gen_logps.max().item())
            # print("[ACTOR DEBUG] Any -inf in gen_logps?", torch.isinf(gen_logps).any().item())
            # print("[ACTOR DEBUG] ans_token_ids min/max:", ans_token_ids.min().item(), ans_token_ids.max().item())
            # print("[ACTOR DEBUG] gen_log_softmax shape:", gen_log_softmax.shape)
            # print("[ACTOR DEBUG] ans_token_ids shape:", ans_token_ids.shape)
            # === END NEW DEBUG PRINTS ===

            entropies = torch.distributions.Categorical(logits=gen_logits_gen_model).entropy()
            print(f"[ACTOR DEBUG] entropies shape: {entropies.shape}")
            print(f"[ACTOR DEBUG] entropies min/max: {entropies.min().item()}, {entropies.max().item()}")
            print(f"[ACTOR DEBUG] entropies has_nan: {torch.isnan(entropies).any().item()}")
            print(f"[ACTOR DEBUG] entropies has_inf: {torch.isinf(entropies).any().item()}")
            # Reference model logps
            # Build attention mask for the full sequence (1 for all tokens, including generated)
            ref_attention_mask = (full_sequences_ids != tokenizer.pad_token_id).long()
            with torch.no_grad():
                ref_logits = model_ref(full_sequences_ids, attention_mask=ref_attention_mask).logits[:, prompt_len - 1 : -1, :]
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
                "prompt_text": msg.get("prompt_text", None),
            })
        else:
            raise RuntimeError(f"Unknown msg type {mtype}")

def generate_rollouts(req_q, resp_q, segment, num_rollouts, item):
    """
    Request the actor to generate rollouts from a given segment prefix.
    Returns a list of dicts with (full_seq, actions, reward, etc.) for each rollout.
    """
    req_q.put({
        "type": "GENERATE",
        "prefix_ids": segment.prefix_ids,
        "num_rollouts": num_rollouts,
        "item": item,
    })
    res = resp_q.get()
    rollouts = []
    for j in range(len(res["full_sequences"])):
        full_seq = res["full_sequences"][j]
        prefix_len = len(segment.prefix_ids)
        gen_part = full_seq[prefix_len:]
        actions = [
            Action(
                gen_part[k],
                res["entropies"][j][k],
                res["gen_logps"][j][k],
                res["ref_logps"][j][k]
            )
            for k in range(len(gen_part))
        ]
        rollouts.append({
            "full_seq": full_seq,
            "actions": actions,
            "reward": res["rewards"][j],
            "gen_logps": res["gen_logps"][j],
            "ref_logps": res["ref_logps"][j],
            "ans_token_ids": gen_part,
            "prefix_len": prefix_len,
            "answer": res["answers"][j],
            "prompt": res.get("prompt_text", None),
        })
    return rollouts


def split_segments(root_seg, item, req_q, resp_q):
    """
    Perform exploitation-only splitting for a fixed number of rounds.
    Returns all rollouts collected during splitting.
    """
    all_rollouts = []
    # TODO eventually 4 will be hyperparam rather than hardcoded
    for round_idx in range(4):
        all_segments = get_all_segments(root_seg)
        leaves = [s for s in all_segments if s.is_leaf and s.length >= 2 * MIN_LEN]
        rewards = [s.mean for s in leaves if s.rollout_count > 0]
        global_mean = sum(rewards) / len(rewards) if rewards else 0.0
        global_std = torch.tensor(rewards).std().item() if len(rewards) > 1 else 1.0
        for s in leaves:
            if s.rollout_count > 0:
                s.advantage = (s.mean - global_mean) / (global_std + 1e-8)
        candidates = [s for s in leaves if s.advantage >= THETA_A]
        if not candidates:
            break
        seg_to_split = max(candidates, key=lambda s: s.advantage)
        split_idx = find_entropy_split(seg_to_split, MIN_LEN)
        if split_idx is None:
            break
        child = seg_to_split.split(split_idx)
        # Generate rollouts from the new segment
        rollouts = generate_rollouts(req_q, resp_q, child, 4, item)
        for rollout in rollouts:
            new_segment = Segment(rollout["actions"], parent=child)
            child.children.append(new_segment)
            new_segment.propagate_reward(rollout["reward"])
            all_rollouts.append({
                "prefix_len": rollout["prefix_len"],
                "sequence": rollout["full_seq"],
                "reward": rollout["reward"],
                "gen_logps": rollout["gen_logps"],
                "ref_logps": rollout["ref_logps"],
                "ans_token_ids": rollout["ans_token_ids"],
                "answer": rollout["answer"],
                "prompt": rollout["prompt"],
            })
    return all_rollouts


def construct_training_batch(all_rollouts, tokenizer):
    """
    Pads and packages the rollouts into a batch dictionary for training.
    """
    sequences, rewards, gen_logps, ref_logps, prompt_lens, ans_token_ids, answers, prompts = [], [], [], [], [], [], [], []
    for traj in all_rollouts:
        sequences.append(traj["sequence"])
        rewards.append(traj["reward"])
        prompt_lens.append(traj["prefix_len"])
        ans_token_ids.append(traj["ans_token_ids"])
        gen_logps.append(traj["gen_logps"])
        ref_logps.append(traj["ref_logps"])
        answers.append(traj.get("answer", ""))
        prompts.append(traj.get("prompt", ""))
    max_seq_len = max(len(s) for s in sequences)
    max_gen_len = max(len(l) for l in gen_logps)
    for s in sequences: s.extend([tokenizer.pad_token_id] * (max_seq_len - len(s)))
    for l in gen_logps: l.extend([-100.0] * (max_gen_len - len(l)))
    for l in ref_logps: l.extend([-100.0] * (max_gen_len - len(l)))
    return {
        "sequences": sequences, "rewards": rewards, "gen_logps": gen_logps,
        "ref_logps": ref_logps, "prompt_lens": prompt_lens,
        "ans_token_ids": ans_token_ids,
        "answers": answers,
        "prompts": prompts,
    }

def collect_training_batch(req_q, resp_q, tokenizer, task_items):
    """
    Orchestrates the full batch collection process for one training step.
    - Samples a prompt
    - Generates initial rollouts
    - Performs exploitation-only splits
    - Packages everything into a training batch
    """
    # Sample a prompt and build the root segment
    item = random.choice(task_items)
    system_prompt = "You are a helpful assistant."
    user_prompt = item["prompt"][0]["content"] if isinstance(item["prompt"], list) else item["prompt"]
    tip_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(tip_text)
    root_seg = Segment([Action(tok, 0.0, 0.0, 0.0) for tok in prompt_tokens])

    # Initial rollouts from root
    initial_rollouts = generate_rollouts(req_q, resp_q, root_seg, 5, item)
    for rollout in initial_rollouts:
        new_segment = Segment(rollout["actions"], parent=root_seg)
        root_seg.children.append(new_segment)
        new_segment.propagate_reward(rollout["reward"])
    all_rollouts = [
        {
            "prefix_len": rollout["prefix_len"],
            "sequence": rollout["full_seq"],
            "reward": rollout["reward"],
            "gen_logps": rollout["gen_logps"],
            "ref_logps": rollout["ref_logps"],
            "ans_token_ids": rollout["ans_token_ids"],
            "answer": rollout["answer"],
            "prompt": rollout["prompt"],
        }
        for rollout in initial_rollouts
    ]

    # Exploitation-only splitting
    all_rollouts += split_segments(root_seg, item, req_q, resp_q)

    # temporary but for let's assert we get expected number of rollouts
    assert len(all_rollouts) == 5 + 4 * 4, f"Expected 21 rollouts, got {len(all_rollouts)}"

    # Batch construction
    batch_raw = construct_training_batch(all_rollouts, tokenizer)
    return batch_raw

def pack_micro_batches(batch_raw, engine, num_micro_batches, micro_batch_size):
    """
    Splits the full batch into micro-batches for gradient accumulation.
    Returns a list of tuples, each containing the tensors and lists for a micro-batch.
    """
    sequences = torch.tensor(batch_raw["sequences"], dtype=torch.long, device=engine.device)
    rewards = torch.tensor(batch_raw["rewards"], dtype=torch.bfloat16, device=engine.device)
    gen_old = torch.tensor(batch_raw["gen_logps"], dtype=torch.bfloat16, device=engine.device)
    ref_logps = torch.tensor(batch_raw["ref_logps"], dtype=torch.bfloat16, device=engine.device)
    prompt_lens = batch_raw["prompt_lens"]
    ans_token_ids = batch_raw["ans_token_ids"]
    # Compute advantages over the full batch
    adv_mean, adv_std = rewards.mean(), rewards.std()
    advantages = (rewards - adv_mean) / (adv_std + 1e-4)
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
            prompt_lens[mb_start:mb_end],
        ))
    return mb_data, rewards, adv_mean, adv_std


def compute_loss_and_step(engine, F, mb_sequences, mb_rewards, mb_gen_old, mb_ref_logps, mb_advantages, mb_ans_token_ids, mb_prompt_lens, step, grad_accum, clip_param, beta, writer=None, rank=0, param_update=0, rewards=None):
    """
    Computes the loss for a micro-batch, performs backward, and steps the optimizer if needed.
    Returns (loss, param_update, should_break) where should_break is True if loss is non-finite.
    """
    logits = engine(mb_sequences).logits
    full_logps = per_token_logps(logits[:, :-1, :], mb_sequences[:, 1:])
    gen_len = mb_gen_old.shape[1]
    curr_logps_list = []
    for i in range(full_logps.shape[0]):
        p_len = mb_prompt_lens[i]
        ans_len = len(mb_ans_token_ids[i])
        sliced = full_logps[i, p_len - 1 : p_len - 1 + ans_len]
        padded = F.pad(sliced, (0, gen_len - ans_len), value=-100.0)
        curr_logps_list.append(padded)
    curr_logps = torch.stack(curr_logps_list, dim=0)

    # Create mask (based on ans_token_ids) to ignore padding BEFORE computing exponentials
    mask = torch.zeros(len(mb_ans_token_ids), gen_len, device=engine.device, dtype=torch.float32)
    for i, toks in enumerate(mb_ans_token_ids):
        mask[i, : len(toks)] = 1.0

    # Clamp differences to prevent overflow in exp()
    ratio_diff = torch.clamp(curr_logps - mb_gen_old.detach(), min=-20.0, max=20.0)
    kl_diff = torch.clamp(mb_ref_logps - curr_logps, min=-20.0, max=20.0)

    # PPO-style loss (same as GRPO) with clamped differences
    ratios = torch.exp(ratio_diff)
    clipped = torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
    pg_loss = -torch.min(ratios * mb_advantages.unsqueeze(-1), clipped * mb_advantages.unsqueeze(-1))
    kl = torch.exp(kl_diff) - kl_diff - 1
    per_tok_loss = pg_loss + beta * kl

    loss = ((per_tok_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)).mean()
    print(f"[DEBUG] loss: {loss.item()}, isfinite={torch.isfinite(loss).item()}")
    if not torch.isfinite(loss):
        if rank == 0:
            print("[Learner][FATAL] Loss became non-finite!", loss.item())
        return loss, param_update, True
    engine.backward(loss)
    # Only update parameters every grad_accum steps
    if ((step + 1) % grad_accum) == 0:
        engine.step()
        grad_norm = engine.get_global_grad_norm()
        param_update += 1
        if writer is not None and rank == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            if rewards is not None:
                writer.add_scalar("train/reward_mean", rewards.mean().item(), step)
                writer.add_scalar("train/reward_std", rewards.std().item(), step)
            print(f"Step {step+1:05d} | ParamUpdate {param_update:05d} | Loss {loss.item():.4f} | Reward {rewards.mean().item() if rewards is not None else 'N/A':.3f}")
    return loss, param_update, False

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
                batch_raw = collect_training_batch(req_q, resp_q, tokenizer, task_items)
            else:
                batch_raw = None
            # Broadcast batch to all ranks
            obj_list = [batch_raw]; dist.broadcast_object_list(obj_list, src=0)
            if rank != 0:
                batch_raw = obj_list[0]
            # Pack micro-batches
            mb_data, rewards, adv_mean, adv_std = pack_micro_batches(batch_raw, engine, num_micro_batches, micro_batch_size)
            if rank == 0 and writer is not None:
                writer.add_scalar("train/reward_mean", rewards.mean().item(), step)
                writer.add_scalar("train/reward_std", rewards.std().item(), step)
                # --- Generation stats logging (gen/) ---
                answers = batch_raw["answers"]
                prompts = batch_raw["prompts"]
                ans_token_ids = batch_raw["ans_token_ids"]
                rewards_tensor = torch.tensor(batch_raw["rewards"])
                completion_lengths = torch.tensor([len(x) for x in ans_token_ids], dtype=torch.float32)
                writer.add_scalar("gen/rewards_mean", rewards_tensor.mean().item(), step)
                writer.add_scalar("gen/rewards_std", rewards_tensor.std().item(), step)
                writer.add_scalar("gen/rewards_min", rewards_tensor.min().item(), step)
                writer.add_scalar("gen/rewards_max", rewards_tensor.max().item(), step)
                writer.add_scalar("gen/completion/mean_length", completion_lengths.mean().item(), step)
                writer.add_scalar("gen/completion/min_length", completion_lengths.min().item(), step)
                writer.add_scalar("gen/completion/max_length", completion_lengths.max().item(), step)
                # --- Per-prompt reward/advantage stats ---
                num_prompts_in_batch = len(set(prompts)) if prompts else 1
                num_rollouts_per_prompt = len(rewards) // num_prompts_in_batch
                rewards_per_prompt = rewards_tensor.view(num_prompts_in_batch, num_rollouts_per_prompt)
                mean_per_prompt = rewards_per_prompt.mean(dim=1, keepdim=True)
                std_per_prompt = rewards_per_prompt.std(dim=1, keepdim=True)
                advantages_per_prompt = (rewards_per_prompt - mean_per_prompt) / (std_per_prompt + 1e-4)
                writer.add_scalar("train/rewards_per_prompt/mean", mean_per_prompt.mean().item(), step)
                writer.add_scalar("train/rewards_per_prompt/std", std_per_prompt.mean().item(), step)
                writer.add_scalar("train/advantages_per_prompt/mean", advantages_per_prompt.mean().item(), step)
                writer.add_scalar("train/advantages_per_prompt/std", advantages_per_prompt.std().item(), step)
                # --- Text table logging ---
                advantages_for_logging = advantages_per_prompt.flatten()
                S = ""
                for p_idx in range(num_prompts_in_batch):
                    prompt_text = prompts[p_idx] if prompts else ""
                    S += f"Prompt: {prompt_text}<br><br>"
                    for r_idx in range(num_rollouts_per_prompt):
                        ans_idx = p_idx * num_rollouts_per_prompt + r_idx
                        answer_text = answers[ans_idx] if answers else ""
                        reward_val = rewards_tensor[ans_idx].item()
                        adv_val = advantages_for_logging[ans_idx].item()
                        S += f"Answer: {answer_text}<br><br>"
                        S += f"Reward: {reward_val}<br><br>"
                        S += f"Advantage: {adv_val}<br><br>"
                writer.add_text("generations", S, step)
        # Now process the next micro-batch
        mb_idx = step % num_micro_batches
        mb_sequences, mb_rewards, mb_gen_old, mb_ref_logps, mb_advantages, mb_ans_token_ids, mb_prompt_lens = mb_data[mb_idx]
        print(f"[DEBUG] mb_sequences shape: {mb_sequences.shape}")
        if mb_sequences.shape[0] == 0:
            print("[WARNING] Skipping empty micro-batch")
            step += 1
            continue
        # Compute loss and step
        loss, param_update, should_break = compute_loss_and_step(
            engine, F,
            mb_sequences, mb_rewards, mb_gen_old, mb_ref_logps, mb_advantages, mb_ans_token_ids, mb_prompt_lens,
            step, grad_accum, clip_param, beta,
            writer=writer if rank == 0 else None, rank=rank, param_update=param_update, rewards=rewards
        )
        # --- Additional logging for ratios, pg_loss, kl stats (match GRPO) ---
        if rank == 0 and writer is not None:
            # Recompute for logging (match compute_loss_and_step)
            with torch.no_grad():
                gen_len = mb_gen_old.shape[1]
                curr_logps_list = []
                for i in range(mb_sequences.shape[0]):
                    p_len = mb_prompt_lens[i]
                    ans_len = len(mb_ans_token_ids[i])
                    sliced = per_token_logps(engine(mb_sequences).logits[:, :-1, :], mb_sequences[:, 1:])[i, p_len - 1 : p_len - 1 + ans_len]
                    padded = F.pad(sliced, (0, gen_len - ans_len), value=-100.0)
                    curr_logps_list.append(padded)
                curr_logps = torch.stack(curr_logps_list, dim=0)
                ratio_diff = torch.clamp(curr_logps - mb_gen_old.detach(), min=-20.0, max=20.0)
                ratios = torch.exp(ratio_diff)
                clipped = torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
                advantages = mb_advantages.unsqueeze(-1)
                pg_loss = -torch.min(ratios * advantages, clipped * advantages)
                kl_diff = torch.clamp(mb_ref_logps - curr_logps, min=-20.0, max=20.0)
                kl = torch.exp(kl_diff) - kl_diff - 1
                writer.add_scalar("train/ratios/mean", ratios.mean().item(), step)
                writer.add_scalar("train/ratios/std", ratios.std().item(), step)
                writer.add_scalar("train/pg_loss/mean", pg_loss.mean().item(), step)
                writer.add_scalar("train/pg_loss/std", pg_loss.std().item(), step)
                writer.add_scalar("train/kl/mean", kl.mean().item(), step)
                writer.add_scalar("train/kl/std", kl.std().item(), step)
        if should_break:
            break
        step += 1

    # Clean up
    if rank == 0:
        req_q.put({"type": "STOP"}); actor.join()


if __name__ == "__main__":
    main() 