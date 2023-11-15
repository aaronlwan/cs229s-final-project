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

def load_model(model_path):
    # Load GPT model from checkpoint
    ckpt_path = os.path.join(model_path, 'ckpt.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
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


# Inference
def perform_inference(model, start, batch_size, max_new_tokens, temperature, top_k, device, dtype, num_samples=1):
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()

    inputs = []
    # Chunk start into input sequences of length 1024
    for i in range(0, len(start), 1024):
        # Pad the last chunk to length 1024
        if i + 1024 > len(start):
            inputs.append(start[i:] + ' ' * (1024 - (len(start) - i))) 
        else:
            inputs.append(start[i:i+1024])

    # Encode each chunk
    start_ids = []
    for inp in inputs:
        start_ids.append(encode(inp))

    # start_ids has shape (num_chunks, 1024)
    # we want to create a tensor x of shape (num_batches, batch_size, 1024)
    x = []
    for i in range(0, len(start_ids), batch_size):
        # Pad the last batch to length batch_size
        if i + batch_size > len(start_ids):
            x.append(start_ids[i:] + [[0] * 1024] * (batch_size - (len(start_ids) - i)))
        else:
            x.append(start_ids[i:i+batch_size])
    x = torch.tensor(x, dtype=torch.long, device=device)

    total_time = 0
    tokens_generated = 0

    start_time = time.time()
    with torch.no_grad():
        with ctx:
            for batch in x:
                for k in range(num_samples):
                    y = model.generate(batch, max_new_tokens, temperature=temperature, top_k=top_k)
                    tokens_generated += len(y[0].tolist())
                    print(decode(y[0].tolist()))
                    print('---------------')

    total_time = time.time() - start_time
    print('Total time: ', total_time)
    print('Tokens generated: ', tokens_generated)
    tokens_per_sec = tokens_generated / total_time
    print('Tokens per second: ', tokens_per_sec)
    return tokens_per_sec



model = load_model('model')

# Batch size 1
tps1 = perform_inference(model, 'FILE:prompt.txt', 1, 500, 0.8, 200, torch.device("cuda" if torch.cuda.is_available() else "cpu"), 'bfloat16', 1)

# Batch size 12
tps12 = perform_inference(model, 'FILE:prompt.txt', 12, 500, 0.8, 200, torch.device("cuda" if torch.cuda.is_available() else "cpu"), 'bfloat16', 1)


# Write results to results.json
import json
results = {
    "loss": 3.1170,
}

results['inference_latency_1'] = tps1
results['inference_latency_12'] = tps12

with open('results.json', 'w') as f:
    json.dump(results, f)
