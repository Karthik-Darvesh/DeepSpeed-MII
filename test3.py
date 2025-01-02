import mii

# DeepSpeed configuration optimized for single GPU (RTX 4070)
ds_config = {
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": False
        },
        "offload_param": {
            "device": "none",
            "pin_memory": False
        }
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "cpu_checkpointing": False
    },
    "tensor_parallel": {
        "enabled": False
    },
    "enable_cuda_graph": False,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0
}

# Deploy the LLaMA model with the latest MII API
mii_config = {
    "task": "text-generation",
    "model": "unsloth/Llama-3.3-70B-Instruct",
    "deployment_name": "llama_70B_single_gpu",
    "deployment_type": "local",
    "enable_deepspeed": True,
    "ds_config": ds_config
}

mii.deploy(**mii_config)

# Query the deployed model
result = mii.query("llama_70B_single_gpu", {"text": "The future of AI is"})
print(result)
