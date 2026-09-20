import torch
from gpt import GPTLanguageModel, encode
import time
device = "cuda" if torch.cuda.is_available() else 'cpu'

model = GPTLanguageModel().to(device).eval()

prompt = torch.tensor(
    encode("Hi there"),
    dtype=torch.long,
    device=device,
).unsqueeze(0)
max_new_tokens = 256


## Verify Correctness
torch.manual_seed(42)
with_cache = model.generate(prompt, max_new_tokens=max_new_tokens, use_kv_cache=True)
torch.manual_seed(42)
without_cache = model.generate(prompt, max_new_tokens=max_new_tokens)

print("Checking Correctness......")
assert torch.equal(with_cache, without_cache)
print("Checking passed")

##Latency and Peaky GPU Memory
warmup_steps = 10
measurement_steps = 20

# Warm-up
for _ in range(warmup_steps):
    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        use_kv_cache=False,
    )

del output
torch.cuda.synchronize()

# Memory already occupied by the model and prompt
baseline_memory = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()

# Measure end-to-end generation latency
start_time = time.perf_counter()

for _ in range(measurement_steps):
    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        use_kv_cache=False,
    )

torch.cuda.synchronize()
elapsed = time.perf_counter() - start_time

peak_memory = torch.cuda.max_memory_allocated()
additional_peak_memory = peak_memory - baseline_memory

print("Without KV Cache")
print(f"Average latency: {elapsed / measurement_steps:.3f} seconds")
print(f"Throughput: {max_new_tokens * measurement_steps / elapsed:.2f} tokens/second")
print(f"Total peak memory: {peak_memory / 1024**2:.2f} MiB")
print(
    f"Additional generation memory: "
    f"{additional_peak_memory / 1024**2:.2f} MiB"
)


warmup_steps = 10
measurement_steps = 20

# Warm-up
for _ in range(warmup_steps):
    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        use_kv_cache=True,
    )

del output
torch.cuda.synchronize()

# Memory already occupied by the model and prompt
baseline_memory = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()

# Measure end-to-end generation latency
start_time = time.perf_counter()

for _ in range(measurement_steps):
    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        use_kv_cache=True,
    )

torch.cuda.synchronize()
elapsed = time.perf_counter() - start_time

peak_memory = torch.cuda.max_memory_allocated()
additional_peak_memory = peak_memory - baseline_memory
print("With KV Cache")
print(f"Average latency: {elapsed / measurement_steps:.3f} seconds")
print(f"Throughput: {max_new_tokens * measurement_steps / elapsed:.2f} tokens/second")
print(f"Total peak memory: {peak_memory / 1024**2:.2f} MiB")
print(
    f"Additional generation memory: "
    f"{additional_peak_memory / 1024**2:.2f} MiB"
)