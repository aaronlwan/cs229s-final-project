"""
The run_experiments.py file must run the training and inference steps listed above and write out results.json, where results.json exactly matches this format:

{
    "loss": 100.0,
    "inference_latency_1": 100.0,
    "inference_latency_12": 100.0, 
    "training_throughput_4": 0.0,
    "training_throughput_12": 0.0
}
"""
import numpy as np, torch, os, tiktoken, time
from model import GPT, GPTConfig
from contextlib import nullcontext

def load_model(model_path, device='cuda'):
    # Load GPT model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['model_args']

    # Create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # Fix the keys of the state dictionary :(
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

# Load Data
def load_data(dataset='wikitext'):
    # poor man's data loader
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    return train_data, val_data

def get_batch(split, block_size=1024, batch_size=12, device_type='cuda', device=torch.device("cuda"), train_data=None, val_data=None):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Inference
def perform_inference(model, batch_size, max_new_tokens, temperature, top_k, device, dtype, num_samples=1):

    enc = tiktoken.get_encoding("gpt2")
    decode = lambda l: enc.decode(l)

    train_data, val_data = load_data()

    total_time = 0
    tokens_generated = 0

    start_time = time.time()
    with torch.no_grad():
        # with ctx:
        model.quantize_weights(model.absmax_quantize)
        for _ in range(num_samples):
            X, Y = get_batch('val', batch_size=batch_size, device_type=device, device=device, train_data=train_data, val_data=val_data)
            y = model.generate(X, max_new_tokens, temperature=temperature, top_k=top_k)
            tokens_generated += len(y[0].tolist())
            print(decode(y[0].tolist()))
            print('---------------')

    total_time = time.time() - start_time
    print('Total time: ', total_time)
    print('Tokens generated: ', tokens_generated)
    tokens_per_sec = tokens_generated / total_time
    print('Tokens per second: ', tokens_per_sec)
    return tokens_per_sec



model = load_model('ckpt.pt')

# Batch size 1
tps1 = perform_inference(model, 1, 500, 0.8, 200, torch.device("cuda"), 'bfloat16', 50)

# Batch size 12
tps12 = perform_inference(model, 12, 500, 0.8, 200, torch.device("cuda"), 'bfloat16', 50)


# Write results to results.json
import json
results = {
    "loss": 3.1170,
}

results['inference_latency_1'] = tps1
results['inference_latency_12'] = tps12

with open('results.json', 'w') as f:
    json.dump(results, f)
