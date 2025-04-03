train_config = {
    "train_batch_size":2,
    "all_steps": 3000,
    "save_steps": 300,
    "wandb_name":"gsm_math_still",
    "wandb_project":"curr_0316",
    "save_path":  "the ckeckpoint path",
    "record_path": "record the generated data during the training process",
    "gen_data_path": "record all the generated data during the training process",
    "gen_device":1,  
    "data_path":"here is the training data path",
    "beta": 0.04,
    "model_path": "here is the base model path",
    "Q_batch_size": 5,
    "num_pre_Q": 8,
    "train_batch_size":2,
    "gen_update_steps": 16,
    "compute_gen_logps": True,
    "clip_param": 0.2,
    "ref_server": "http://localhost:59807",
    "port": 59807,
    "wandb_key":"here is your wandb key"
}

ds_config = {
    "train_micro_batch_size_per_gpu": train_config['train_batch_size'],
    "gradient_accumulation_steps": 4,
    "steps_per_print": 5,
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