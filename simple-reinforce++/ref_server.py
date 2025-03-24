
import json, os, shutil, re, random, io, time
import torch
from config import base_config, ds_config
from transformers import AutoTokenizer
get_num = 0
macro_step = base_config["train_gpu_num"] * ds_config["gradient_accumulation_steps"]
model_path = base_config["model_path"]
beta = base_config["beta"]
tokenizer = AutoTokenizer.from_pretrained(base_config["model_path"])

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()
def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)
def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()
def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

def batch_standardize_padding(batch, mask):
    batch_valid = [adv[mask.bool()] for adv, mask in zip(batch, mask)]
    batch_valid_tensor = torch.cat([r.flatten() for r in batch_valid])
    mean = batch_valid_tensor.mean().item()
    std = batch_valid_tensor.std(unbiased=False).item() + 1e-5
    return mean, std
  
    
def get_valid_num(batch):
    valid_num=0
    for item in batch:
        valid_num+=item.sum().item()

    return valid_num

def get_eos(completion_mask):
    seq_len = completion_mask.size(1)

    reversed_mask = torch.flip(completion_mask, dims=[1])
    last_indices_rev = reversed_mask.argmax(dim=1)  
    last_indices = seq_len - 1 - last_indices_rev  

    eos_mask = torch.zeros_like(completion_mask)
    eos_mask.scatter_(1, last_indices.unsqueeze(1), 1)
    return eos_mask


if __name__ == '__main__':   
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn as nn

    from bottle import request
    import bottle, threading, queue
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    ref_model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
    ref_model.eval()
    ref_model.requires_grad_(False)

    def get_per_token_logps(input_ids):
        logits = ref_model(input_ids).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    raw_queue = queue.Queue()
    result_queue = queue.Queue()

    app = bottle.Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        dd = request.body.read()
        dd = bytes_list_to_list(dd)

        data = {'base': json.loads(dd[0])} 
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        if len(dd) == 4: data['gen_logps'] = bytes_to_tensor(dd[3])
        raw_queue.put(data)
        print('receive', data['inputs'].shape, data['rewards'], 
              data['gen_logps'].shape if 'gen_logps' in data else '')
        return b'tensor'

    @app.route('/get', method='GET')
    def do_get():
        global get_num
        if result_queue.empty(): return b'empty'
        else:
            data=result_queue.get()
            get_num += 1
            # Only 3 latest macro batches are stored in the queue
            if get_num == macro_step:
                while result_queue.qsize() // macro_step > 3:
                    for i in range(macro_step):
                        result_queue.get() 
                get_num = 0
        return data
    
    def run_server(): bottle.run(app, host='0.0.0.0', port=base_config["port"], server='tornado')
    threading.Thread(target=run_server, daemon=False).start()

    while True:
        all_data=[]
        all_batch_advantages=[]
        all_batch_reward=[]
        all_completion_mask=[]
        # Calculate the reward for macro batch
        for i in range(macro_step):
            d = raw_queue.get()
            prompt_length = d['base']['plen']
            with torch.inference_mode():
                ref_per_token_logps = get_per_token_logps(d['inputs'].to(ref_model.device))
            ref_per_token_logps = ref_per_token_logps[:,prompt_length-1:].to('cpu')
            d["ref_logps"] = ref_per_token_logps

            all_data.append(d)
            
            gen_per_token_logps = d['gen_logps']
            completion_mask = (d['inputs'][:, prompt_length:] != tokenizer.pad_token_id).int()

            eos_mask = get_eos(completion_mask)

            per_token_kl = gen_per_token_logps - ref_per_token_logps.to(gen_per_token_logps.device) 
            per_token_reward = d['rewards'].unsqueeze(1).expand(per_token_kl.shape) * eos_mask
            per_token_reward = (per_token_reward - beta * per_token_kl) * completion_mask
            
            all_batch_reward.append(per_token_reward)
            all_completion_mask.append(completion_mask)
        
        # Reward standardization
        reward_mean, reward_std = batch_standardize_padding(all_batch_reward, all_completion_mask)

        for i,item in enumerate(all_data):
            reward = all_batch_reward[i]
            reward = (reward-reward_mean)/reward_std
            completion_mask = all_completion_mask[i]
            reward = reward * completion_mask
            per_token_advantage = torch.flip(torch.cumsum(torch.flip(reward, dims=(1,)), dim=1), dims=(1,)) 
            all_batch_advantages.append(per_token_advantage)

        # Advantage standardization
        adv_mean, adv_std = batch_standardize_padding(all_batch_advantages, all_completion_mask)

        # Get the number of valid tokens in the macro batch
        valid_num = get_valid_num(all_completion_mask)

        for i,item in enumerate(all_data):
            advantages = all_batch_advantages[i]
            advantages = (advantages - adv_mean) / adv_std
            item['base']['num_items_in_batch'] = valid_num / macro_step
            data = [json.dumps(item['base']).encode(), tensor_to_bytes(item['inputs']), 
                    tensor_to_bytes(item['rewards']), tensor_to_bytes(item['ref_logps']),tensor_to_bytes(item['gen_logps']),tensor_to_bytes(advantages)]
            xdata = make_bytes_list(data)
            result_queue.put(xdata)
