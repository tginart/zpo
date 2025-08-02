from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct, math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union, List, Tuple, Dict
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model_path = "Qwen/Qwen2.5-1.5B-instruct"
gen_device = 1          # physical GPU that vLLM should use
beta = 0.04
all_steps = 1000
Q_batch_size = 1 # In ASPO, we process one prompt at a time to build a tree
num_pre_Q = 1 # Not really used in ASPO in the same way
train_batch_size = 8
gen_update_steps = 16
save_steps = 200
compute_gen_logps = False # ASPO handles logps differently
clip_param = 0.2
ref_server = "http://localhost:59875"

# ASPO specific hyperparameters
max_rollouts = 16
k_rollouts = 4
theta_A = 1.0
min_len = 8
high_temp = 1.0
max_new_tokens = 700
temperature = 0.9

from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0])
    data['prompt_ids'] = bytes_to_tensor(dd[1])
    data['prompt_mask'] = bytes_to_tensor(dd[2])
    data['completion_ids'] = bytes_to_tensor(dd[3])
    data['completion_mask'] = bytes_to_tensor(dd[4])
    data['advantages'] = bytes_to_tensor(dd[5])
    if beta > 0:
        data['refs'] = bytes_to_tensor(dd[6])
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def ASPO_step(batch):
    prompt_ids = batch['prompt_ids'].to(engine.device)
    prompt_mask = batch['prompt_mask'].to(engine.device)
    completion_ids = batch['completion_ids'].to(engine.device)
    completion_mask = batch['completion_mask'].to(engine.device)
    advantages = batch['advantages'].to(engine.device)

    inputs = torch.cat([prompt_ids, completion_ids], dim=1)
    
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]
    
    # We only care about logps for completion tokens
    max_completion_len = completion_ids.size(1)
    logits_for_completion = logits[:, -max_completion_len-1:-1, :]
    
    per_token_logps = get_per_token_logps(logits_for_completion, completion_ids)
    
    # Detached logps for ratio calculation (since we don't have old_logps from generator)
    old_per_token_logps = per_token_logps.detach()
    ratio = torch.exp(per_token_logps - old_per_token_logps)
    clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
    
    per_token_loss = -torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1))

    if beta > 0:
        ref_per_token_logps = batch['refs'].to(per_token_logps.device)
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        per_token_loss += beta * per_token_kl

    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()
    return loss


@dataclass
class Action:
    token_id: int
    entropy: float

class ListSegment:
    def __init__(
        self,
        actions: List[Action],
        *,
        parent: Optional["ListSegment"] = None,
        initial_rollout_count: int = 0,
        initial_mean: float = 0.0,
        initial_m2: float = 0.0,
    ) -> None:
        self.actions: List[Action] = actions
        self.parent: Optional["ListSegment"] = parent
        self.children: List["ListSegment"] = []
        self.rollout_count: int = initial_rollout_count
        self._mean: float = initial_mean
        self._m2: float = initial_m2

    def update_stats(self, reward: float):
        self.rollout_count += 1
        delta = reward - self._mean
        self._mean += delta / self.rollout_count
        delta2 = reward - self._mean
        self._m2 += delta * delta2

    @property
    def value(self) -> float:
        return self._mean if self.rollout_count > 0 else 0.0

    @property
    def variance(self) -> float:
        if self.rollout_count < 2: return 0.0
        return self._m2 / self.rollout_count

    @property
    def length(self) -> int: return len(self.actions)

    def split(self, split_idx: int) -> "ListSegment":
        child_actions = self.actions[split_idx:]
        self.actions = self.actions[:split_idx]
        child = ListSegment(
            child_actions, parent=self,
            initial_rollout_count=self.rollout_count,
            initial_mean=self._mean, initial_m2=self._m2,
        )
        self.children.append(child)
        return child

    def collect_token_ids(self) -> List[int]:
        chain: List["ListSegment"] = []
        cur: Optional["ListSegment"] = self
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        token_ids: List[int] = []
        for seg in reversed(chain):
            token_ids.extend(a.token_id for a in seg.actions)
        return token_ids

class RootSegment(ListSegment):
    def __init__(self, actions: List[Action]):
        super().__init__(actions, parent=None)

def gen_worker(Q, physics_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {physics_device}")
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.5, max_model_len=2048)

    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
    
    system_prompt = """You are a helpful assistant...<think> reasoning process here </think><answer> answer here </answer>."""

    from math_verify import parse, verify, ExprExtractionConfig
    def reward_fn(text):
        item_q, answer = text.split(tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=False)) # hacky way to get Q
        item_q = item_q.replace(tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}, {"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True)[:-1], "")
        item = next((item for item in QAs if item['Q'] == item_q), None)
        if item is None: return -2.0

        pattern = r'\d+\.\d+|\d+/\d+|\d+'
        nums = re.findall(pattern, answer) 
        if len(nums) == 0: correct_reward = -1.0
        else:
            lastnum = nums[-1]
            ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
            ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
            correct_reward = 1 if verify(ans, ground_truth) else -1
        
        pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
        think_count = answer.count("<think>") + answer.count("</think>")
        answer_count = answer.count("<answer>") + answer.count("</answer>")
        format_reward = 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2 else -1
        return correct_reward + format_reward

    def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        pd = torch.softmax(logits, dim=-1)
        return torch.sum(-pd * torch.log(pd + 1e-9), dim=-1)

    def _generate_from_segment(segment: ListSegment, temp: float):
        token_ids = segment.collect_token_ids()
        prompt_text = tokenizer.decode(token_ids)
        num_prefix_tokens = len(token_ids)
        
        sampling_params = SamplingParams(temperature=temp, top_p=0.9, max_tokens=max_new_tokens - num_prefix_tokens, logprobs=1)
        if (max_new_tokens - num_prefix_tokens) <=0: return [], []
        
        voutputs = vllm_gen.generate([prompt_text], sampling_params, use_tqdm=False)
        output = voutputs[0].outputs[0]
        
        new_token_ids = output.token_ids
        
        entropies = []
        if output.logprobs:
            for logprob_step in output.logprobs:
                logits = torch.tensor(list(logprob_step.values())[0].logprob)
                entropies.append(_entropy_from_logits(logits).item())
        
        return new_token_ids, entropies

    def _propagate_reward(segment: ListSegment, reward: float):
        cur = segment
        while cur is not None:
            cur.update_stats(reward)
            cur = cur.parent

    def _iter_segments(root: RootSegment):
        stack = [root]
        while stack:
            seg = stack.pop()
            yield seg
            stack.extend(reversed(seg.children))

    def _global_mean_var(root: RootSegment):
        vals = [s.value for s in _iter_segments(root) if s.rollout_count > 0]
        if not vals: return 0.0, 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return mean, var

    def find_entropy_split(seg: ListSegment, min_len: int):
        if seg.length < 2 * min_len: return None
        
        special_ids = set(tokenizer.all_special_ids)
        candidates = []
        for idx in range(min_len, seg.length - min_len):
            if seg.actions[idx].token_id in special_ids: continue
            candidates.append({"index": idx, "entropy": seg.actions[idx].entropy})

        if not candidates: return None
        best_candidate = max(candidates, key=lambda x: x["entropy"])
        return {"best_idx": best_candidate["index"], "entropy_at_split": best_candidate["entropy"]}

    def argmax_branch_se(root: RootSegment):
        max_se, best = float("-inf"), None
        _, global_var = _global_mean_var(root)

        for seg in _iter_segments(root):
            visited_children = [c for c in seg.children if c.rollout_count > 0]
            k = len(visited_children)
            if k < 2: continue
            
            sum_of_terms = sum((c.variance if c.rollout_count >= 2 else global_var) / c.rollout_count for c in visited_children)
            se = math.sqrt(max(sum_of_terms, 0)) / k
            if se > max_se: max_se, best = se, seg
        return best

    def _do_rollouts(start_seg: ListSegment, k: int, temp: float):
        for _ in range(k):
            new_token_ids, entropies = _generate_from_segment(start_seg, temp)

            if not new_token_ids:
                full_text = tokenizer.decode(start_seg.collect_token_ids())
                reward = reward_fn(full_text)
                _propagate_reward(start_seg, reward)
                continue

            new_actions = [Action(tok, ent) for tok, ent in zip(new_token_ids, entropies)]
            child_seg = ListSegment(new_actions, parent=start_seg)
            start_seg.children.append(child_seg)
            
            full_text = tokenizer.decode(child_seg.collect_token_ids())
            reward = reward_fn(full_text)
            _propagate_reward(child_seg, reward)

    def run_episode(root: RootSegment):
        rollouts = 0
        while rollouts < max_rollouts:
            # Exploit
            candidates: List[Tuple[ListSegment, float]] = []
            global_mean, global_var = _global_mean_var(root)
            global_std = math.sqrt(max(global_var, 1e-8))

            for seg in _iter_segments(root):
                if seg.length < 2 * min_len: continue
                adv = (seg.value - global_mean) / global_std if global_std > 0 else 0.0
                if adv >= theta_A: candidates.append((seg, adv))
            
            candidates.sort(key=lambda x: x[1], reverse=True)

            split_done = False
            for seg, adv in candidates:
                split_info = find_entropy_split(seg, min_len)
                if split_info is None: continue
                
                child = seg.split(split_info["best_idx"])
                _do_rollouts(seg, k_rollouts, temperature)
                rollouts += k_rollouts
                split_done = True
                break

            if split_done: continue

            # Explore
            branch = argmax_branch_se(root) or root
            _do_rollouts(branch, k_rollouts, high_temp)
            rollouts += k_rollouts

    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print('[VLLM PROC] recving new model ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print('[VLLM PROC] model updated')
            del new_state_dict
        except:
            return

    from torch.nn.utils.rnn import pad_sequence
    for it in range(999999999):
        if it % 3 == 0: try_update_model()
        
        prompt_item = random.choice(QAs)
        prompt_text = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_item['Q']}], tokenize=False, add_generation_prompt=True)
        
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        root = RootSegment([Action(tok, 0.0) for tok in prompt_ids])

        run_episode(root)

        g_mean, g_var = _global_mean_var(root)
        g_std = math.sqrt(max(g_var, 1e-8))

        collected_segments = [seg for seg in _iter_segments(root) if seg.rollout_count > 0 and seg.length > 0 and seg is not root]
        
        if not collected_segments: continue

        for i in range(0, len(collected_segments), train_batch_size):
            batch_segments = collected_segments[i:i+train_batch_size]
            
            advantages = torch.tensor([(s.value - g_mean) / g_std if g_std > 0 else 0.0 for s in batch_segments], dtype=torch.float32)
            
            prompt_ids_list, completion_ids_list = [], []
            for seg in batch_segments:
                full_ids = seg.collect_token_ids()
                prefix_len = len(full_ids) - seg.length
                prompt_ids_list.append(torch.tensor(full_ids[:prefix_len]))
                completion_ids_list.append(torch.tensor(full_ids[prefix_len:]))

            prompt_ids_padded = pad_sequence(prompt_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
            completion_ids_padded = pad_sequence(completion_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
            prompt_mask = (prompt_ids_padded != tokenizer.pad_token_id).long()
            completion_mask = (completion_ids_padded != tokenizer.pad_token_id).long()

            data = [
                json.dumps({"info": "aspo_batch"}).encode(),
                tensor_to_bytes(prompt_ids_padded),
                tensor_to_bytes(prompt_mask),
                tensor_to_bytes(completion_ids_padded),
                tensor_to_bytes(completion_mask),
                tensor_to_bytes(advantages),
            ]
            if beta > 0:
                # ref logps calculation is needed here.
                # for simplicity, we pass zeros for now.
                ref_logps = torch.zeros_like(completion_ids_padded, dtype=torch.float32)
                data.append(tensor_to_bytes(ref_logps))

            xdata = make_bytes_list(data)
            requests.post(f"{ref_server}/upload", data=xdata)
        
        print(f"Finished episode for prompt: {prompt_item['Q'][:50]}... Collected {len(collected_segments)} segments. Global mean reward: {g_mean:.3f}")


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()

    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, gen_device))
        p.start()

    model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa")

    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                                model_parameters=model.parameters())

    progress = range(1, all_steps+1)
    if dist.get_rank() == 0: progress = tqdm(progress)
    for step in progress:
        batch = get_batch()
        while batch is None:
            if dist.get_rank() == 0: print('waiting for batch...'); time.sleep(1)
            else: time.sleep(0.2)
            batch = get_batch()

        loss = ASPO_step(batch)
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")

        if step % gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                Q.put(state_dict)
                print('[TRAINING PROC] send state_dict ok!')
            dist.barrier()

        if step % save_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"./step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier() 