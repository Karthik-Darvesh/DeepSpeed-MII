import os
import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Multinode Inference Example")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--hostfile", type=str, default="/home/karthik22/Deepspeed_testing/DeepSpeed-MII/hostfile", help="Path to hostfile")
    parser.add_argument("--port", type=str, default="29500", help="Port for distributed communication")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    return parser.parse_args()

def setup_distributed(hostfile, port):
    # Read hostfile to get rank and world size
    with open(hostfile, 'r') as f:
        hosts = f.readlines()
    world_size = sum([int(line.strip().split('slots=')[1]) for line in hosts])
    
    # Get the current node rank
    current_node = os.environ['NODE_RANK'] if 'NODE_RANK' in os.environ else '0'
    current_rank = int(current_node) * 2  # Assuming 2 slots per node for simplicity

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method=f'tcp://{hosts[0].strip().split()[0]}:{port}', world_size=world_size, rank=current_rank)
    torch.cuda.set_device(current_rank % torch.cuda.device_count())

def main():
    args = parse_args()
    
    # Initialize distributed training
    setup_distributed(args.hostfile, args.port)
    
    # Initialize DeepSpeed
    ds_config = {
        "train_batch_size": args.batch_size,
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        }
    }
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Prepare DeepSpeed for inference
    model_engine, _, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=None,
        training_data=None
    )
    
    # Wrap the model with DDP
    model_engine = DDP(model_engine, device_ids=[model_engine.local_rank], output_device=model_engine.local_rank)
    
    # Example inference
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(model_engine.device)
    
    with torch.no_grad():
        outputs = model_engine.module.generate(**inputs, max_length=50)
    
    if model_engine.local_rank == 0:
        print("Generated Text:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
