"""
Distributed synchronous ASPO with:
  • DeepSpeed learner on N GPUs (data-parallel ranks)
  • Generation actor on *one* dedicated GPU that is **not** part of the DeepSpeed
    visibility set.  Example launch command:

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
deepspeed --include localhost:0,1,2,3,4,5,6 \
          aspo_vllm_sync_ds_multi.py \
          --actor_gpu 7  # Note: do NOT pass --num_gpus when using --include

The conductor (rank 0) runs the actor in a separate process and manages the
ASPO search tree. It sends prefixes to the actor for rollout generation and
receives back completed sequences with rewards and other metadata. After an
ASPO episode, it assembles a training batch and broadcasts it to all learner
ranks.

NOTE: The PPO loss calculation remains largely the same as in GRPO, but the
batch is constructed from the diverse set of trajectories generated during the
ASPO search process.
"""

import os, random, time, json, argparse, math
from typing import List, Dict, Any, Optional, Tuple

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
# ASPO Data Structures
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
    def __init__(self, actions: List[Action], parent: Optional['Segment'] = None):
        self.actions = actions
        self.parent = parent
        self.children: List['Segment'] = []
        
        # Running reward statistics (updated via Welford's algorithm)
        self.rollout_count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squares of differences from the mean

        # ASPO metrics calculated at the start of each generation loop
        self.advantage = 0.0
        self.branch_se = 0.0

    @property
    def variance(self) -> float:
        """Returns the sample variance. Returns 0 if less than 2 rollouts."""
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
    def prefix_ids(self) -> List[int]:
        """Gathers all token IDs from self and ancestors to form the full prefix."""
        if self.parent:
            return self.parent.prefix_ids + [a.token_id for a in self.actions]
        return [a.token_id for a in self.actions]
    
    @property
    def full_logps(self) -> List[float]:
        if self.parent:
            return self.parent.full_logps + [a.logp for a in self.actions]
        return [a.logp for a in self.actions]
    
    @property
    def full_ref_logps(self) -> List[float]:
        if self.parent:
            return self.parent.full_ref_logps + [a.ref_logp for a in self.actions]
        return [a.ref_logp for a in self.actions]

    def split(self, split_idx: int) -> 'Segment':
        """
        Splits the segment at `split_idx`. The original segment is truncated,
        and a new child segment is created with the suffix of actions. The child
        inherits the parent's full statistical state.
        """
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
        """Updates statistics of this segment and all its ancestors using Welford's algorithm."""
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

# ----------------------------------------------------------------------------------
# Hyper-parameters (can also be overridden by CLI args)
# ----------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_gpu", type=int, default=1, help="Global GPU id reserved for the generation actor")
    parser.add_argument("--local_rank", type=int, default=-1, help="(internal) local rank passed by DeepSpeed")
    parser.add_argument("--train_gpus", type=int, default=7, help="Number of training devices")

    # Core training params
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-instruct")
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--num_prompts", type=int, default=2, help="Number of prompts to process in each policy update step")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_gen_tokens", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-6)

    # PPO loss params
    parser.add_argument("--beta", type=float, default=0.001, help="KL penalty coefficient")
    parser.add_argument("--clip_param", type=float, default=0.1)    

    # ASPO params
    parser.add_argument("--k", type=int, default=4, help="Number of rollouts per split/explore step")
    parser.add_argument("--min_ro", type=int, default=12, help="Min rollouts before considering early exit from explore mode")
    parser.add_argument("--max_ro", type=int, default=20, help="Max rollouts before forcing bonus rollouts")
    parser.add_argument("--theta_a", type=float, default=1.0, help="Advantage threshold for triggering a split")
    parser.add_argument("--min_len", type=int, default=16, help="Minimum length for a segment to be split or for its child")

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

args = get_args()

# Static params -----------------------------------------------------------
task = args.task
model_path = args.model_path
# Force num_prompts to 1 as per user requirement - ASPO works with single data point
num_prompts = 1
gradient_accumulation_steps = args.gradient_accumulation_steps
max_gen_tokens = args.max_gen_tokens
beta = args.beta
clip_param = args.clip_param
lr = args.lr
num_train_devices = args.train_gpus # Hardcoded as per user request

# ASPO params
K = args.k
MIN_RO = args.min_ro
MAX_RO = args.max_ro
THETA_A = args.theta_a
MIN_LEN = args.min_len

# The total number of rollouts per prompt is now dynamic due to ASPO logic.
# The train_batch_size will be determined by the number of sequences collected
# during an ASPO episode. We assume it will be divisible by the number of training devices.
# A check will be added later before broadcasting the batch.

# DeepSpeed runtime config ------------------------------------------------
DS_CONFIG: Dict[str, Any] = {
    # train_micro_batch_size_per_gpu is fixed at 4 as per user requirement
    "train_micro_batch_size_per_gpu": 4,
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
    "gradient_clipping": 1.0,
}

# ----------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------

def per_token_logps(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return torch.gather(logp, -1, ids.unsqueeze(-1)).squeeze(-1)

def get_all_segments(root: Segment) -> List[Segment]:
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

    # The valid split indices range from `min_len` to `length - min_len`
    for i in range(min_len, segment.length - min_len + 1):
        # We look at the entropy of the action *at* index i-1, which is the
        # last action of the proposed truncated parent.
        entropy = segment.actions[i - 1].entropy
        if entropy > max_entropy:
            max_entropy = entropy
            split_idx = i
            
    return split_idx if split_idx != -1 else None

def argmax_branch_se(root: Segment, global_variance: float) -> Optional[Segment]:
    """
    Finds the node in the tree with the highest Branch Standard Error (SE),
    which is a measure of uncertainty among its children.
    """
    all_segments = get_all_segments(root)
    branches = [s for s in all_segments if len(s.children) >= 2]
    
    if not branches:
        return None

    max_se = -1.0
    best_branch = None

    for branch in branches:
        # Only consider children that have been evaluated (i.e., they are roots of rollouts)
        evaluated_children = [c for c in branch.children if c.rollout_count > 0]
        if len(evaluated_children) < 2:
            branch.branch_se = 0
            continue

        sum_of_variances = 0
        for child in evaluated_children:
            # Use child's empirical variance if available, else use global variance
            child_var = child.variance if child.rollout_count >= 2 else global_variance
            sum_of_variances += child_var / child.rollout_count
        
        k = len(evaluated_children)
        branch.branch_se = (1 / k) * math.sqrt(sum_of_variances)

        if branch.branch_se > max_se:
            max_se = branch.branch_se
            best_branch = branch
            
    return best_branch

# ----------------------------------------------------------------------------------
# Actor process (runs only once, launched by rank-0)
# ----------------------------------------------------------------------------------

def actor_process(actor_gpu: int, req_q: mp.Queue, resp_q: mp.Queue, max_gen_tokens: int):
    # make chosen GPU the only visible one for this process BEFORE importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(actor_gpu)
    import torch  # re-import inside new context

    # The actor is on a single GPU, so it becomes cuda:0 inside this process
    actor_device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Use separate models for generation and reference
    model_gen = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(actor_device)
    model_ref = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(actor_device)
    model_gen.eval()
    model_ref.eval()

    # Inform the learner that the actor is fully initialized
    resp_q.put({"status": "READY"})

    _, reward_fn = get_task(task)

    while True:
        msg = req_q.get()
        mtype = msg["type"]
        if mtype == "STOP":
            break
        elif mtype == "UPDATE_WEIGHTS":
            model_gen.load_state_dict(msg["state_dict"])
            resp_q.put({"status": "OK"})
        elif mtype == "GENERATE":
            tic = time.time()
            
            prefix_ids = msg["prefix_ids"]
            num_rollouts = msg["num_rollouts"]
            temperature = msg["temperature"]
            reward_context = msg["reward_context"]

            inputs = torch.tensor([prefix_ids], device=actor_device)

            # Generate sequences
            outputs = model_gen.generate(
                inputs,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                max_new_tokens=max_gen_tokens,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=num_rollouts,
                output_scores=True,
                return_dict_in_generate=True,
            )
            full_sequences_ids = outputs.sequences
            
            # Decode answers (completion only)
            prompt_len = len(prefix_ids)
            ans_token_ids = full_sequences_ids[:, prompt_len:]
            answers = tokenizer.batch_decode(ans_token_ids, skip_special_tokens=True)

            # Calculate rewards
            rewards = reward_fn(answers, items=[reward_context] * len(answers))

            # --- Calculate logps and entropies for the generated part ---
            
            # Logits for generated tokens are in `outputs.scores`
            # Stack them: (num_tokens, batch_size, vocab_size) -> (batch_size, num_tokens, vocab_size)
            gen_logits_gen_model = torch.stack(outputs.scores, dim=1)
            
            # Get logps from the generation model
            gen_log_softmax = torch.log_softmax(gen_logits_gen_model, dim=-1)
            gen_logps = torch.gather(gen_log_softmax, -1, ans_token_ids.unsqueeze(-1)).squeeze(-1)

            # Calculate entropy from the generation model's logits
            softmax = torch.softmax(gen_logits_gen_model, dim=-1)
            log_softmax = torch.log_softmax(gen_logits_gen_model, dim=-1)
            entropies = -(softmax * log_softmax).sum(dim=-1)

            # Get logps from the reference model
            with torch.no_grad():
                ref_logits = model_ref(full_sequences_ids).logits[:, prompt_len - 1 : -1, :]
            ref_log_softmax = torch.log_softmax(ref_logits, dim=-1)
            ref_logps = torch.gather(ref_log_softmax, -1, ans_token_ids.unsqueeze(-1)).squeeze(-1)
            
            gen_time = time.time() - tic

            resp_q.put({
                "full_sequences": full_sequences_ids.cpu().tolist(),
                "rewards": rewards,
                "gen_logps": gen_logps.cpu().tolist(),
                "ref_logps": ref_logps.cpu().tolist(),
                "entropies": entropies.cpu().tolist(),
                "gen_time": gen_time,
            })
        else:
            raise RuntimeError(f"Unknown msg type {mtype}")

# ----------------------------------------------------------------------------------
# Conductor/Learner Helper Functions (run on rank 0)
# ----------------------------------------------------------------------------------

def get_all_segments(root: Segment) -> List[Segment]:
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

    # The valid split indices range from `min_len` to `length - min_len`
    for i in range(min_len, segment.length - min_len + 1):
        # We look at the entropy of the action *at* index i-1, which is the
        # last action of the proposed truncated parent.
        entropy = segment.actions[i - 1].entropy
        if entropy > max_entropy:
            max_entropy = entropy
            split_idx = i
            
    return split_idx if split_idx != -1 else None

def argmax_branch_se(root: Segment, global_variance: float) -> Optional[Segment]:
    """
    Finds the node in the tree with the highest Branch Standard Error (SE),
    which is a measure of uncertainty among its children.
    """
    all_segments = get_all_segments(root)
    branches = [s for s in all_segments if len(s.children) >= 2]
    
    if not branches:
        return None

    max_se = -1.0
    best_branch = None

    for branch in branches:
        # Only consider children that have been evaluated (i.e., they are roots of rollouts)
        evaluated_children = [c for c in branch.children if c.rollout_count > 0]
        if len(evaluated_children) < 2:
            branch.branch_se = 0
            continue

        sum_of_variances = 0
        for child in evaluated_children:
            # Use child's empirical variance if available, else use global variance
            child_var = child.variance if child.rollout_count >= 2 else global_variance
            sum_of_variances += child_var / child.rollout_count
        
        k = len(evaluated_children)
        branch.branch_se = (1 / k) * math.sqrt(sum_of_variances)

        if branch.branch_se > max_se:
            max_se = branch.branch_se
            best_branch = branch
            
    return best_branch

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

    # Queues and Actor process only on the Conductor (rank-0)
    if rank == 0:
        mp.set_start_method("spawn", force=True)
        req_q, resp_q = mp.Queue(), mp.Queue()
        actor = mp.Process(
            target=actor_process,
            args=(args.actor_gpu, req_q, resp_q, max_gen_tokens),
        )
        actor.start()
    else:
        req_q = resp_q = None  # type: ignore

    # Tokenizer & model (same on all ranks)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    # Defer setting train_micro_batch_size_per_gpu until we know the batch size from ASPO
    engine, optimizer, _, _ = deepspeed.initialize(
        config=DS_CONFIG,
        model=model,
        model_parameters=model.parameters(),
    )

    print(f'[Rank {rank}] DeepSpeed engine created.')

    if rank == 0:
        writer = SummaryWriter(f"runs/aspo_sync_multi_{time.strftime('%Y%m%d-%H%M%S')}")
        # log the args as text
        # (logging args is good practice, keeping it concise for brevity)
        writer.add_text("args", json.dumps(vars(args)))
        writer.add_text("algo", 'ASPO')


    # ------------------------------------------------------------------
    # Wait until the actor process has finished loading its models and
    # signals readiness, then synchronize all ranks.
    # ------------------------------------------------------------------
    if rank == 0:
        ready_msg = resp_q.get()  # blocks until actor sends a READY status
        if ready_msg.get("status") != "READY":
            raise RuntimeError("Actor failed to initialize properly")

    # Ensure every rank waits until the actor is ready
    dist.barrier()
    
    task_items, _ = get_task(task)

    # Start training loop
    for step in trange(1, args.steps + 1, desc="Training Steps"):
        
        # ========================================================================
        # == STAGE 1: CONDUCTOR (RANK 0) RUNS ASPO TO GENERATE A BATCH ==========
        # ========================================================================
        if rank == 0:
            print(f"\n[Conductor] Step {step}: Starting ASPO episode...")
            
            # --- 1. Send latest weights to actor ---
            state_dict_cpu = {k: v.to("cpu") for k, v in engine.module.state_dict().items()}
            req_q.put({"type": "UPDATE_WEIGHTS", "state_dict": state_dict_cpu})
            resp_q.get() # wait for ACK

            # --- 2. Initialize ASPO roots and storage for this step's rollouts ---
            prompt_items = random.sample(task_items, num_prompts)
            aspo_roots = []
            system_prompt = "You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it."
            for item in prompt_items:
                user_prompt = item["prompt"][0]["content"] if isinstance(item["prompt"], list) else item["prompt"]
                tip_text = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_tokens = tokenizer.encode(tip_text)
                # We start with dummy actions for the root so it has content
                root_actions = [Action(tok, 0.0, 0.0, 0.0) for tok in prompt_tokens]
                aspo_roots.append({"root": Segment(root_actions), "item": item, "rollouts_done": 0})

            all_generated_trajectories = [] # This will store all data for the final training batch

            # --- 3. Run the ASPO generation loop for each prompt ---
            for i, proot in enumerate(aspo_roots):
                root_seg = proot["root"]
                
                # Initial rollouts from the root prompt
                print(f"[Conductor]   Prompt {i}: Initial rollouts...")
                req_q.put({
                    "type": "GENERATE",
                    "prefix_ids": root_seg.prefix_ids,
                    "num_rollouts": K,
                    "temperature": 0.9,
                    "reward_context": proot["item"],
                })
                res = resp_q.get()
                
                # Process initial rollouts
                for j in range(len(res["full_sequences"])):
                    full_seq = res["full_sequences"][j]
                    gen_part = full_seq[root_seg.length:]
                    actions = [Action(gen_part[k], res["entropies"][j][k], res["gen_logps"][j][k], res["ref_logps"][j][k]) for k in range(len(gen_part))]
                    
                    new_segment = Segment(actions, parent=root_seg)
                    root_seg.children.append(new_segment)
                    new_segment.propagate_reward(res["rewards"][j])
                    
                    # Store for training batch
                    all_generated_trajectories.append({
                        "prefix_len": root_seg.length,
                        "sequence": full_seq,
                        "reward": res["rewards"][j],
                        "gen_logps": new_segment.full_logps[root_seg.length:],
                        "ref_logps": new_segment.full_ref_logps[root_seg.length:],
                    })
                proot["rollouts_done"] += K

                # --- ASPO Main Loop (Exploit/Explore) ---
                is_exploring = False
                while proot["rollouts_done"] < MAX_RO:
                    all_segments = get_all_segments(root_seg)
                    if not all_segments: break
                    
                    # Recalculate global stats and advantages
                    all_rewards = [s.mean for s in all_segments if s.rollout_count > 0]
                    global_mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0
                    global_std = torch.tensor(all_rewards).std().item() if len(all_rewards) > 1 else 1.0
                    global_variance = global_std**2

                    for s in all_segments:
                        if s.rollout_count > 0:
                            s.advantage = (s.mean - global_mean) / (global_std + 1e-8)
                        else:
                            raise RuntimeError("Segment has no rollouts")
                            s.advantage = 0
                    
                    # -- 1. EXPLOIT --
                    candidates = [s for s in all_segments if s.advantage >= THETA_A and s.length >= 2 * MIN_LEN and s.is_leaf]
                    candidates.sort(key=lambda s: s.advantage, reverse=True)

                    split_done = False
                    if candidates:
                        seg_to_split = candidates[0]
                        split_idx = find_entropy_split(seg_to_split, MIN_LEN)
                        if split_idx is not None:
                            print(f"[Conductor]   Prompt {i}: Splitting segment (adv={seg_to_split.advantage:.2f})")
                            child = seg_to_split.split(split_idx)
                            
                            # Do rollouts from the *newly truncated parent*
                            req_q.put({
                                "type": "GENERATE", "prefix_ids": seg_to_split.prefix_ids, "num_rollouts": K,
                                "temperature": 0.9, "reward_context": proot["item"],
                            })
                            res = resp_q.get()
                            proot["rollouts_done"] += K
                            split_done = True
                            is_exploring = False

                            # Process results: new segments become siblings to the one created by the split
                            for j in range(len(res["full_sequences"])):
                                full_seq = res["full_sequences"][j]
                                gen_part = full_seq[seg_to_split.length:]
                                actions = [Action(gen_part[k], res["entropies"][j][k], res["gen_logps"][j][k], res["ref_logps"][j][k]) for k in range(len(gen_part))]
                                new_segment = Segment(actions, parent=seg_to_split)
                                seg_to_split.children.append(new_segment)
                                new_segment.propagate_reward(res["rewards"][j])
                                all_generated_trajectories.append({
                                    "prefix_len": seg_to_split.length,
                                    "sequence": full_seq,
                                    "reward": res["rewards"][j],
                                    "gen_logps": [a.logp for a in actions],
                                    "ref_logps": [a.ref_logp for a in actions]
                                })


                    if split_done:
                        continue
                    
                    # -- 2. EXPLORE --
                    is_exploring = True
                    if proot["rollouts_done"] >= MIN_RO:
                        break # Will proceed to bonus rollouts

                    explore_branch = argmax_branch_se(root_seg, global_variance) or root_seg
                    print(f"[Conductor]   Prompt {i}: Exploring from branch (SE={explore_branch.branch_se:.4f})")
                    
                    req_q.put({
                        "type": "GENERATE", "prefix_ids": explore_branch.prefix_ids, "num_rollouts": K,
                        "temperature": 1.0, "reward_context": proot["item"],
                    })
                    res = resp_q.get()
                    proot["rollouts_done"] += K
                    
                    # Process results
                    for j in range(len(res["full_sequences"])):
                        full_seq = res["full_sequences"][j]
                        gen_part = full_seq[explore_branch.length:]
                        actions = [Action(gen_part[k], res["entropies"][j][k], res["gen_logps"][j][k], res["ref_logps"][j][k]) for k in range(len(gen_part))]
                        new_segment = Segment(actions, parent=explore_branch)
                        explore_branch.children.append(new_segment)
                        new_segment.propagate_reward(res["rewards"][j])
                        all_generated_trajectories.append({
                            "prefix_len": explore_branch.length,
                            "sequence": full_seq,
                            "reward": res["rewards"][j],
                            "gen_logps": [a.logp for a in actions],
                            "ref_logps": [a.ref_logp for a in actions]
                        })

                # --- 4. Bonus Rollouts to get to a multiple of num_train_devices (7) ---
                target_rollouts = 0
                if is_exploring: # Was in explore mode when we finished
                    target_rollouts = math.ceil(proot["rollouts_done"] / num_train_devices) * num_train_devices
                else: # Was in exploit mode
                    target_rollouts = math.ceil(MAX_RO / num_train_devices) * num_train_devices
                
                bonus_needed = target_rollouts - proot["rollouts_done"]
                if bonus_needed > 0:
                    print(f"[Conductor]   Prompt {i}: Doing {bonus_needed} bonus rollouts (target={target_rollouts})...")
                    bonus_branch = argmax_branch_se(root_seg, 1.0) or root_seg # Use global_variance=1.0 if not calculated this iter
                    req_q.put({
                        "type": "GENERATE", "prefix_ids": bonus_branch.prefix_ids, "num_rollouts": bonus_needed,
                        "temperature": 1.0, "reward_context": proot["item"],
                    })
                    res = resp_q.get()
                    # Process bonus results
                    for j in range(len(res["full_sequences"])):
                         full_seq = res["full_sequences"][j]
                         gen_part = full_seq[bonus_branch.length:]
                         actions = [Action(gen_part[k], res["entropies"][j][k], res["gen_logps"][j][k], res["ref_logps"][j][k]) for k in range(len(gen_part))]
                         new_segment = Segment(actions, parent=bonus_branch)
                         bonus_branch.children.append(new_segment)
                         new_segment.propagate_reward(res["rewards"][j])
                         all_generated_trajectories.append({
                             "prefix_len": bonus_branch.length,
                             "sequence": full_seq,
                             "reward": res["rewards"][j],
                             "gen_logps": [a.logp for a in actions],
                             "ref_logps": [a.ref_logp for a in actions]
                         })
            
            # --- 5. Finalize and package batch for all learners ---
            total_sequences = len(all_generated_trajectories)
            if total_sequences % num_train_devices != 0:
                 # This should ideally not happen due to bonus rollouts, but as a safeguard:
                 print(f"Warning: Total sequences ({total_sequences}) not divisible by num_train_devices ({num_train_devices}). This may cause issues.")

            # Keep micro batch size fixed at 4 as per user requirement
            
            # Collate data
            sequences, rewards, gen_logps, ref_logps, prompt_lens, ans_token_ids = [], [], [], [], [], []
            for traj in all_generated_trajectories:
                sequences.append(traj["sequence"])
                rewards.append(traj["reward"])
                prompt_lens.append(traj["prefix_len"])
                
                # Extract logps from Action objects
                ans_token_ids.append([t for t in traj["sequence"][traj["prefix_len"]:]])
                gen_logps.append(traj["gen_logps"])
                ref_logps.append(traj["ref_logps"])
                
            # Pad all lists to max length for tensor conversion
            max_seq_len = max(len(s) for s in sequences)
            max_gen_len = max(len(l) for l in gen_logps)

            for s in sequences: s.extend([tokenizer.pad_token_id] * (max_seq_len - len(s)))
            for l in gen_logps: l.extend([0.0] * (max_gen_len - len(l)))
            for l in ref_logps: l.extend([0.0] * (max_gen_len - len(l)))
            
            batch = {
                "sequences": sequences, "rewards": rewards, "gen_logps": gen_logps,
                "ref_logps": ref_logps, "prompt_lens": prompt_lens,
                "ans_token_ids": ans_token_ids, # Used for masking
            }

        else: # Other ranks
            batch = None

        # ========================================================================
        # == STAGE 2: ALL RANKS RECEIVE BATCH AND DO A TRAINING STEP =============
        # ========================================================================
        
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
        # This part is tricky with ASPO's dynamic rollouts. A simpler approach is
        # to normalize all advantages together.
        adv_mean = rewards.mean()
        adv_std = rewards.std()
        advantages = (rewards - adv_mean) / (adv_std + 1e-4)
        
        if rank == 0:
            writer.add_scalar("train/rewards/mean", rewards.mean().item(), step)
            writer.add_scalar("train/rewards/std", rewards.std().item(), step)
            writer.add_scalar("train/advantages/mean", advantages.mean().item(), step)
            writer.add_scalar("train/advantages/std", advantages.std().item(), step)

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
            print(f"Step: {step:05d} | Loss: {loss.item():.4f} | Mean Reward: {rewards.mean().item():.3f} | Batch Size: {len(sequences)}")
            writer.add_scalar("train/loss_step", loss.item(), step)
            writer.add_scalar("train/reward_step", rewards.mean().item(), step)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm, step)
            
            # Could add more detailed logging here if needed

    # cleanup
    if rank == 0:
        req_q.put({"type": "STOP"})
        actor.join()


if __name__ == "__main__":
    main() 