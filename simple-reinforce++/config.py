base_config={
  "model_path": "/mnt/remote-data/downloads/models/Qwen/Qwen2-7B",
  "gen_device": "1", # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES
  "train_gpu_num": 2, # Number of GPUs used for training
  "train_batch_size": 4,
  "beta": 0.01,
  "all_steps": 1200, 
  "Q_batch_size": 32,
  "num_pre_Q": 1,
  "gen_update_steps": 16,
  "save_steps": 200,
  "clip_param": 0.2,
  "port":51414,
  "ref_server": "http://localhost:51414",
  
}


ds_config = {
    "train_micro_batch_size_per_gpu": base_config["train_batch_size"],
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    # "gradient_clipping": 1,
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
