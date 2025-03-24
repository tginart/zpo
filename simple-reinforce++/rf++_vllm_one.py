from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct, traceback
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from config import ds_config, base_config
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
model_path = base_config["model_path"]
gen_device = base_config["gen_device"]  # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES
beta = base_config["beta"]
all_steps = base_config["all_steps"]
Q_batch_size = base_config["Q_batch_size"]
num_pre_Q = base_config["num_pre_Q"]
train_batch_size = base_config["train_batch_size"]
gen_update_steps = base_config["gen_update_steps"]
save_steps = base_config["save_steps"]
clip_param = base_config["clip_param"]
ref_server = base_config["ref_server"]


from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list


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
    if len(dd) >= 5: 
        data['gen_logps'] = bytes_to_tensor(dd[4])
    if len(dd)==6: 
        data["advantages"] = bytes_to_tensor(dd[5])
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def REINFORCE_plusplus_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    per_token_advantage = batch['advantages'].to(engine.device)
    gen_per_token_logps=batch['gen_logps'].to(engine.device)
    num_items_in_batch = batch['num_items_in_batch']
    
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]

    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    ratio = torch.exp(per_token_logps - gen_per_token_logps)
    clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
    per_token_loss = -torch.min(ratio * per_token_advantage, clipped_ratio * per_token_advantage)

    loss = (per_token_loss * completion_mask).sum() / num_items_in_batch
    return loss

def gen_worker(Q, physics_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {physics_device}")
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.7)
    ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload

    sampling_params = SamplingParams(n=num_pre_Q, temperature=0.7, max_tokens=650)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    # from datasets import load_dataset
    # data_path = "/mnt/remote-data/hjy/public_code/GRPO/data/gsm8k_train.json"
    # with open(data_path, 'r', encoding='utf-8') as file:
    #     dataset = json.load(file)
    # QAs = [{'Q': item['question'], 'A': item['answer_only']} for item in dataset]
    data_path = "/mnt/remote-data/hjy/data/o1/math/train/MATH_train-cleaned_processed.json"
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    QAs = [{'Q': item['question'], 'A': item['answer_detail']} for item in dataset]

    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
    The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here. Please ensure that your final answer is enclosed within \\boxed{} </answer>."""
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

    from math_verify import parse, verify, ExprExtractionConfig

    def reward_correct(ground_truth, answer):
        pattern = r'\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}'
        boxed_matches_ans =  re.findall(pattern, answer)
        boxed_matches_ground_truth =  re.findall(pattern, ground_truth)
        if not boxed_matches_ans or not boxed_matches_ground_truth:
            return False
        boxed_matches_ans = "\\boxed{" + boxed_matches_ans[-1] + "}"
        boxed_matches_ground_truth =  "\\boxed{" + boxed_matches_ground_truth[-1] + "}"

        if not boxed_matches_ans:
            return False
        
        if boxed_matches_ans == ground_truth:
            return True

        ans = parse(boxed_matches_ans)
        print(ans)
        ground_truth = parse(boxed_matches_ground_truth)
        print(ground_truth)
        return True if verify(ans, ground_truth) else False
    
    def reward_format(answer):
        pattern = r"^<think>.*?</think>[\n ]<answer>.*?</answer>$"
        think_count = answer.count("<think>") + answer.count("</think>")
        answer_count = answer.count("<answer>") + answer.count("</answer>")
        return True if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count == 2 and answer_count == 2 else False
    
    def gen_samples(inputs):
        prompts = [x["Q"] for x in inputs]
        answers, ans_token_ids = gen_answers(prompts)
        rewards = []
        for i, inp in enumerate(inputs):
            for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
                if reward_format(a) and reward_correct(inp['A'],a):
                    reward=2
                elif reward_format(a) and not reward_correct(inp['A'],a):
                    reward=0.5
                else:
                    reward=-2
                rewards.append(reward)
        prompts_text = [tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
        return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids

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
        if it % 2 == 0: try_update_model()
        inputs = random.sample(QAs, Q_batch_size)
        tic = time.time()
        prompt_inputs, rewards, answers, ans_token_ids = gen_samples(inputs)
        print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards, )
        if it % 5 == 0: 
            print('answers:', answers[0])
            os.makedirs("./result", exist_ok=True)
            with open("./result/answer.txt", 'a') as file:
                file.write(f"--------------------\n{answers[0]}\n")

        if rewards.max() - rewards.min() < 1e-4: continue
        all_batch_num = Q_batch_size * num_pre_Q
        qa_idx_list = list(range(all_batch_num))
        random.shuffle(qa_idx_list)
        for ii in range(0, all_batch_num, train_batch_size):
            batch_indices = qa_idx_list[ii:ii+train_batch_size]
            
            sub_prompts = [prompt_inputs[idx//num_pre_Q] for idx in batch_indices]
            sub_ans_ids = [ans_token_ids[idx] for idx in batch_indices]
            sub_rewards = rewards[batch_indices]

            Qrep = tokenizer(sub_prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)["input_ids"]

            plen = Qrep.shape[1]
            tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
            output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
            merged_ids = torch.cat([Qrep, output_ids], dim=1)                        
            data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]       

            zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
            zz = [xx.prompt_logprobs[plen:] for xx in zz]
            gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
            data.append(tensor_to_bytes(gen_logps))

            xdata = make_bytes_list(data)
            r = requests.post(f"{ref_server}/upload", data=xdata)

               
tokenizer = AutoTokenizer.from_pretrained(model_path)
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
    wating_flag = True
    for step in progress:
        batch = get_batch()
        while batch is None:
            if wating_flag:
                print('waiting for batch...')
                wating_flag=False
            time.sleep(3)
            batch = get_batch()
        wating_flag = True

        loss = REINFORCE_plusplus_step(batch)
        
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