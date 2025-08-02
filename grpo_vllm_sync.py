"""
A synchronous (single-process) version of the GRPO script in `grpo_vllm.py`.

Changes with regard to the original asynchronous implementation
---------------------------------------------------------------
1. No HTTP ref_server, no multiprocessing, no DeepSpeed queue.
   A single Python process does everything in the following order on every
   training step:
       - sample prompts
       - generate completions using the current policy
       - compute rewards via `tasks.get_task`
       - compute reference & generation log-probs
       - calculate GRPO loss and update the policy

2. Uses 🤗 Transformers `generate` instead of vLLM.  This is slower but keeps
   the control flow simple and avoids another CUDA context.

3. Still relies on the same reward & task interface (`tasks.get_task`).  The
   KL penalty, PPO clipping, and β hyper-parameter are preserved so the loss
   should match the original when run with the same inputs.

The script is intentionally minimal so that it is easier to set break-points
and print intermediate tensors while debugging.

Usage
-----
$ python simple_GRPO/grpo_vllm_sync.py

The code is *not* tuned for speed – it is aimed at correctness and
readability.
"""

import os, random, time, json
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter
from vllm import LLM, SamplingParams  # add vLLM for fast generation

from tasks import get_task  # same helper as the async script

# ---------------------------
# Hyper-parameters (copied)
# ---------------------------

task = "length"
model_path = "Qwen/Qwen2.5-1.5B-instruct"

all_steps = 1000
train_batch_size = 8            # number of completions used for a backward() call
num_prompts_per_step = 1                  # use a single prompt per step for constant prompt length
num_generations_per_prompt = train_batch_size  # completions per prompt so that num_prompts * num_generations == train_batch_size
assert num_prompts_per_step * num_generations_per_prompt == train_batch_size

beta = 0.04     # KL penalty weight
clip_param = 0.2
lr = 1e-6
max_gen_tokens = 128

# ---------------------------
# Helpers
# ---------------------------

torch.backends.cuda.matmul.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)

# ---------------------------
# vLLM inference engine (policy)
# ---------------------------

# We use a separate vLLM engine for sampling completions. After every optimiser
# step we will push the updated PyTorch weights into the vLLM runner so that
# generation stays on-policy.

llm_gen = LLM(model=model_path, gpu_memory_utilization=0.45)

system_prompt = (
    """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
)

# Sampling parameters for the main generation pass (completions)
sampling_params = SamplingParams(
    n=num_generations_per_prompt,
    temperature=0.9,
    top_p=0.9,
    max_tokens=max_gen_tokens,
)

# Sampling parameters for obtaining per-token log-probs (prompt_logprobs)
gen_logps_sp = SamplingParams(
    temperature=0,
    top_p=1,
    max_tokens=1,
    prompt_logprobs=1,
)

# policy (trainable)  -----------------
policy = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(device)

policy_lr = torch.optim.AdamW(policy.parameters(), lr=lr)

# reference (frozen) -------------------
reference = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=policy.dtype,
).to(device)
reference.eval()
for p in reference.parameters():
    p.requires_grad_(False)

# task prompts & reward fn -------------
all_items, reward_fn = get_task(task)

# TensorBoard --------------------------
run_id = f"grpo_sync_{time.strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(f"runs/{run_id}")

# ---------------------------
# Utility functions
# ---------------------------

def get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return log-probs of the *given* tokens under the provided logits.

    logits: (B, L, V)
    input_ids: (B, L)
    returns: (B, L)  – log p(token_t | context < t)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def sample_batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Generate a batch using vLLM and compute rewards / log-probs.

    Returns
    -------
    merged_ids : (B, L) int64  – prompt + completion ids (padded)
    rewards    : (B,)          float32
    gen_logps  : (B, L_c)      float32 – log p under *policy* at generation time
    prompt_len : int           – number of tokens in the prompt
    """
    # ---- choose one task item and extract user prompt -------------------
    item = random.choice(all_items)
    if isinstance(item["prompt"], list):
        for msg in item["prompt"]:
            if msg["role"] == "user":
                user_prompt = msg["content"]
                break
    else:
        user_prompt = item["prompt"]

    # Repeat the same prompt so that batch_size == train_batch_size --------
    prompts_expanded = [user_prompt] * train_batch_size

    # Build chat-formatted prompts accepted by Qwen tokenizer -------------
    tip_text = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts_expanded
    ]

    # ---- vLLM generation -------------------------------------------------
    v_outputs = llm_gen.generate(tip_text, sampling_params, use_tqdm=False)

    answers: List[str] = []
    ans_token_ids: List[List[int]] = []
    for v in v_outputs:
        for o in v.outputs:
            answers.append(o.text)
            ans_token_ids.append(o.token_ids)

    # ---- convert to padded tensor --------------------------------------
    from torch.nn.utils.rnn import pad_sequence

    tensor_list = [torch.tensor(t, dtype=torch.long) for t in ans_token_ids]
    merged_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)

    # prompt length (same for every example) ------------------------------
    prompt_len = len(tokenizer(tip_text[0], add_special_tokens=False)["input_ids"])

    # ---- rewards --------------------------------------------------------
    rewards_list = reward_fn(answers, items=[item] * len(answers))
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

    # ---- per-token log-probs at generation time -------------------------
    zz = llm_gen.generate(
        prompt_token_ids=merged_ids.tolist(),
        sampling_params=gen_logps_sp,
        use_tqdm=False,
    )
    gen_logps_rows: List[List[float]] = []
    for z in zz:
        per_tok = [list(t.values())[0].logprob for t in z.prompt_logprobs[prompt_len:]]
        gen_logps_rows.append(per_tok)

    # pad to equal length --------------------------------------------------
    max_len = max(len(r) for r in gen_logps_rows)
    for r in gen_logps_rows:
        r.extend([0.0] * (max_len - len(r)))
    gen_logps = torch.tensor(gen_logps_rows, dtype=torch.float32, device=device)

    return merged_ids, rewards, gen_logps, prompt_len


# ---------------------------
# GRPO loss (same maths)
# ---------------------------

def grpo_loss(batch_sequences: torch.Tensor,
              rewards: torch.Tensor,
              gen_logps_old: torch.Tensor,
              prompt_len: int) -> torch.Tensor:
    """Compute GRPO loss for a batch (synchronous version).

    Parameters
    ----------
    batch_sequences : (B, L) int64  – prompt + completion ids, padded
    rewards         : (B,)          – scalar rewards (pre-normalised)
    gen_logps_old   : (B, L_c)      – log p at generation time (completion part)
    prompt_len      : int           – token count of the prompts
    """
    # ---- forward through *current* policy ------------------------------
    logits = policy(batch_sequences).logits  # (B, L, V)
    per_token_logps = get_per_token_logps(logits[:, :-1, :], batch_sequences[:, 1:])
    per_token_logps = per_token_logps[:, prompt_len - 1 :]  # completion only

    # ---- reference log-probs -------------------------------------------
    with torch.no_grad():
        ref_logits = reference(batch_sequences).logits
    ref_logps = get_per_token_logps(ref_logits[:, :-1, :], batch_sequences[:, 1:])
    ref_logps = ref_logps[:, prompt_len - 1 :]

    # ---- PPO ratio ------------------------------------------------------
    ratios = torch.exp(per_token_logps.detach() - gen_logps_old)  # stop-grad current logps here
    clipped_ratios = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param)

    advantages = rewards.unsqueeze(-1)  # broadcast along tokens
    pg_loss = -torch.min(ratios * advantages, clipped_ratios * advantages)

    # ---- KL penalty -----------------------------------------------------
    per_token_kl = torch.exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1.0

    per_token_loss = pg_loss + beta * per_token_kl

    # ---- mask padding ---------------------------------------------------
    completion_mask = (batch_sequences[:, prompt_len:] != tokenizer.pad_token_id).float()
    masked_loss = (per_token_loss * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-8)

    return masked_loss.mean()


# ---------------------------
# Training loop
# ---------------------------

for step in range(1, all_steps + 1):
    policy.train()

    sequences, rewards, gen_logps_old, prompt_len = sample_batch()

    # standardise rewards per prompt group (as the async code does)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    loss = grpo_loss(sequences, rewards, gen_logps_old, prompt_len)

    policy_lr.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    policy_lr.step()

    # Keep the vLLM engine in sync with the updated weights so that the next
    # call to `sample_batch()` reflects the current policy.
    llm_gen.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
        policy.state_dict().items()
    )

    # --- logging --------------------------------------------------------
    if step % 10 == 0:
        print(f"step {step:>5} | loss {loss.item():.4f} | reward μ {rewards.mean().item():.3f}")
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/reward_mean", rewards.mean().item(), step)

    # (optional) save checkpoints
    if step % 200 == 0:
        ckpt_dir = f"sync_step_{step}"
        os.makedirs(ckpt_dir, exist_ok=True)
        policy.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

print("Training finished") 