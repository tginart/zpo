from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

task = "length"
model_path = "Qwen/Qwen2.5-1.5B-instruct"
gen_device = 1          # physical GPU that vLLM should use
beta = 0.04
all_steps = 1000
Q_batch_size = 5
num_pre_Q = 8
train_batch_size = 8
gen_update_steps = 1
save_steps = 200
compute_gen_logps = True
lr = 1e-6
clip_param = 0.2
ref_server = "http://localhost:59875"
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

# Import task system
from tasks import get_task

# Load task data and reward function
task_items, task_reward_fn = get_task(task)

ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": gen_update_steps,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": lr }
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
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    if compute_gen_logps:
        assert len(dd) == 5, "Mismatch in data packet size when expecting generation log probabilities."
        data['gen_logps'] = bytes_to_tensor(dd[4])
    else:
        assert len(dd) == 4, "Mismatch in data packet size."
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
#from kernel.ce_kernel import fast_log_softmax_gather
#get_per_token_logps = fast_log_softmax_gather

def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    # GRPO loss combines a PPO-like policy gradient term with a KL divergence penalty.
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    
    if 'gen_logps' in batch:
        # PPO-style clipped objective
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        # Fallback to REINFORCE-style objective if old log probabilities are not available
        # pg_loss = -per_token_logps * advantages
        # for debugging, let's crash here
        raise ValueError("gen_logps not found in batch")

    # Combine policy gradient loss and KL penalty
    per_token_loss = pg_loss + beta * per_token_kl

    # Masked and normalized loss
    masked_loss = (per_token_loss * completion_mask).sum(dim=1)
    sum_of_mask = completion_mask.sum(dim=1)
    
    # Avoid division by zero for empty completions
    loss_per_sequence = masked_loss / sum_of_mask
    loss = loss_per_sequence.nan_to_num().mean()
    return loss


def gen_worker(Q, physics_device, run_id=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {physics_device}")
    writer = None
    if run_id:
        writer = SummaryWriter(f"runs/{run_id}")

    from vllm import LLM, SamplingParams
    # Policy model (gets updated) and reference model (static)
    vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.45)
    vllm_ref = LLM(model=model_path, gpu_memory_utilization=0.45, enforce_eager=True) # Enforce eager to avoid sharing weights
    ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload

    sampling_params = SamplingParams(n=num_pre_Q, temperature=0.9, max_tokens=1000)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    # Load task data and reward function
    task_items, task_reward_fn = get_task(task)
    
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
    The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
    def gen_answers(prompts):
        tip_text = []
        for x in prompts:
            tip_text.append(tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
        voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
        answers = [];  ans_token_ids = []
        for v in voutputs:
            for z in v.outputs: 
                answers.append(z.text)
                ans_token_ids.append(z.token_ids)
        return answers, ans_token_ids

    gen_table_columns = ["prompt", "generation", "reward", "advantage"]

    def gen_samples(inputs):
        # Extract prompts from task items
        prompts = []
        for x in inputs:
            if isinstance(x["prompt"], list):
                # Convert chat format to string
                prompt_text = ""
                for msg in x["prompt"]:
                    if msg["role"] == "user":
                        prompt_text = msg["content"]
                        break
                prompts.append(prompt_text)
            else:
                prompts.append(x["prompt"])
        
        answers, ans_token_ids = gen_answers(prompts)
        
        # Use task reward function
        rewards = task_reward_fn(answers, items=inputs)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        prompts_text = [tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
        return prompts_text, rewards, answers, ans_token_ids

    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print('[VLLM PROC] recving new model ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print('[VLLM PROC] model updated')
            del new_state_dict
        except:
            #print('[VLLM PROC] no new model')
            return
        
    from torch.nn.utils.rnn import pad_sequence
    for it in range(999999999):
        try_update_model()
        inputs = random.sample(task_items, Q_batch_size)
        tic = time.time()
        prompt_inputs, rewards, answers, ans_token_ids = gen_samples(inputs)
        gen_time = time.time()-tic
        print(f'time: {gen_time:.2f}s    ', 'rewards:', rewards, )

        if writer:
            try:
                completion_lengths = torch.tensor([len(x) for x in ans_token_ids], dtype=torch.float32)
                rewards_grouped = rewards.view(-1, num_pre_Q)
                std_grouped_rewards = rewards_grouped.std(dim=1)
                is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))
                frac_reward_zero_std = is_std_zero.float().mean().item()
                
                log_dict = {
                    "gen/time": gen_time,
                    "gen/rewards_mean": rewards.mean().item(),
                    "gen/rewards_std": rewards.std().item(),
                    "gen/rewards_min": rewards.min().item(),
                    "gen/rewards_max": rewards.max().item(),
                    "gen/completion/mean_length": completion_lengths.mean().item(),
                    "gen/completion/min_length": completion_lengths.min().item(),
                    "gen/completion/max_length": completion_lengths.max().item(),
                    "gen/reward_std_per_group": std_grouped_rewards.mean().item(),
                    "gen/frac_reward_zero_std": frac_reward_zero_std,
                }
                for k, v in log_dict.items():
                    writer.add_scalar(k, v, it)
            except Exception as e:
                print(f"[gen_worker] tensorboard logging failed: {e}", file=sys.stderr)
                raise  # Re-raise the exception

        #if it % 5 == 0: print('answers:', answers[0])

        if writer: #and it % 20 == 0:
            try:
                # Reshape rewards and calculate advantages for table logging
                rewards_grouped = rewards.view(Q_batch_size, num_pre_Q)
                
                mean_grouped_rewards = rewards_grouped.mean(dim=1, keepdim=True)
                std_grouped_rewards = rewards_grouped.std(dim=1, keepdim=True)
                advantages = (rewards_grouped - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
                
                def escape_for_markdown_table(s):
                    return s.replace("\n", "<br/>").replace("|", "\\|")

                # Only show 4 columns: prompt, generation, reward, and advantage
                table_str = "| Prompt | Generation | Reward | Advantage |\n"
                table_str += "| --- | --- | --- | --- |\n"
                for i in range(len(inputs)):
                    # Extract prompt content
                    if isinstance(inputs[i]['prompt'], list):
                        prompt = inputs[i]['prompt'][0]['content']
                    else:
                        prompt = inputs[i]['prompt']
                    
                    for j in range(num_pre_Q):
                        p_text = escape_for_markdown_table(prompt)
                        a_text = escape_for_markdown_table(answers[i * num_pre_Q + j])
                        r_val = rewards_grouped[i, j].item()
                        adv_val = advantages[i, j].item()
                        table_str += f"| {p_text} | {a_text} | {str(int(r_val))} | {str(round(adv_val, 2))} |\n"
                        #writer.add_text("generations/prompt", prompt, it)
                        #writer.add_text("generations/completion", answers[i * num_pre_Q + j], it)
                        #writer.add_text("generations/reward", f"{r_val:.4f}", it)
                        #writer.add_text("generations/advantage", f"{adv_val:.4f}", it)

                writer.add_text("generations", table_str, it)

            except Exception as e:
                print(f"[gen_worker] tensorboard add_text failed: {e}", file=sys.stderr)
                raise  # Re-raise the exception

        for i, pp in enumerate(prompt_inputs):
            prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
            plen = prompt_ids.shape[1]
            curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
            if curr_rewards.max() - curr_rewards.min() < 1e-4: continue

            if ref_server_ver == 'tensor':
                curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                for ii in range(0, num_pre_Q, train_batch_size):
                    sub_rewards = curr_rewards[ii:ii+train_batch_size]
                    sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                    tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                    output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
                    Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                    merged_ids = torch.cat([Qrep, output_ids], dim=1)

                    # Get reference log probabilities from the static reference model
                    ref_outputs = vllm_ref.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                    ref_logps_list = [o.prompt_logprobs[plen:] for o in ref_outputs]
                    ref_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in ref_logps_list])
                    
                    data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards), tensor_to_bytes(ref_logps)]       

                    if compute_gen_logps:
                        zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                        zz = [xx.prompt_logprobs[plen:] for xx in zz]
                        gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                        data.append(tensor_to_bytes(gen_logps))

                    xdata = make_bytes_list(data)
                    r = requests.post(f"{ref_server}/upload", data=xdata)
                    if r.content == b'string': ref_server_ver = 'string'
            elif ref_server_ver == 'string':
                xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(), 
                                        tensor_to_bytes(curr_rewards)])
                r = requests.post(f"{ref_server}/upload", data=xdata)
                if r.content == b'tensor': ref_server_ver = 'tensor'


tokenizer = AutoTokenizer.from_pretrained(model_path)
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()

    run_id = None
    writer = None
    if dist.get_rank() == 0:
        run_id = f"grpo_{time.strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(f"runs/{run_id}")
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, gen_device, run_id))
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
            print('waiting for batch...'); time.sleep(1)
            batch = get_batch()

        loss = GRPO_step(batch)
        print(f"loss: {loss.item()}")
        raise ValueError("test")
        exit()
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")
            if writer:
                writer.add_scalar("train/loss", loss.item(), step)
        raise ValueError("test")
        exit()
            #else:
            #    raise ValueError("writer is None")

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
