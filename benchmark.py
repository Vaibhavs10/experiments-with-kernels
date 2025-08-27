import os
import gc
import time
import statistics
from dataclasses import dataclass, field

import torch
from torch.utils import benchmark
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config


# ----------------------------
# Config
# ----------------------------
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
DEVICE = "cuda"                # Single H100
SEED = 1234
NUM_THREADS = torch.get_num_threads()
TOKEN_BUDGETS = [512, 2048]
BATCH_SIZES = [1, 16, 32]       # batch sizes to test (processes multiple inputs simultaneously)
WARMUP_GENERATIONS = 1           # warmup calls (not timed)
MEM_REPEATS = 3                  # times to measure peak memory per setting
TIMER_MIN_RUNTIME_S = 2.0        # torch.benchmark blocked_autorange budget

# Attention implementations to benchmark
ATTN_IMPLEMENTATIONS = [
    "eager",
    "sdpa", 
    "kernels-community/flash-attn3:flash_attention",
    "kernels-community/flash-attn"
]


# ----------------------------
# Utilities
# ----------------------------
def set_determinism(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_gib(x_bytes: int) -> float:
    return x_bytes / (1024 ** 3)


def clear_cuda():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


@dataclass
class Point:
    latency_s: float
    tokens: int
    toks_per_s: float
    peak_alloc_gib: float
    peak_reserved_gib: float
    batch_size: int = 1


@dataclass
class ScenarioResult:
    label: str
    batch_size: int = 1
    # map: max_new_tokens -> Point
    by_tokens: dict = field(default_factory=dict)


# ----------------------------
# Model + inputs
# ----------------------------
def load_model(attn_implementation: str):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map=DEVICE,
        attn_implementation=attn_implementation,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Ensure tokenizer has a pad token for proper batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, model.device


def build_inputs(tokenizer, device, batch_size=1):
    # Create different prompts for each item in the batch to simulate realistic usage
    base_prompts = [
        "What is Tensor Parallelism?",
        "Explain machine learning fundamentals.",
        "How do neural networks work?",
        "What are the benefits of distributed computing?",
        "Describe the attention mechanism in transformers.",
        "What is gradient descent?",
        "How does backpropagation work?",
        "Explain the concept of overfitting.",
    ]
    
    # Cycle through prompts to create a batch
    batch_messages = []
    for i in range(batch_size):
        prompt = base_prompts[i % len(base_prompts)]
        messages = [{"role": "system", "content": prompt}]
        batch_messages.append(messages)
    
    # Apply chat template to each conversation in the batch
    batch_texts = []
    for messages in batch_messages:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        batch_texts.append(text)
    
    # Tokenize all texts together with padding to handle variable lengths
    if batch_size == 1:
        inputs = tokenizer(
            batch_texts[0],
            return_tensors="pt",
            return_dict=True,
        )
        return inputs.to(device)
    else:
        # Use padding to handle variable-length sequences
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            return_dict=True,
            padding=True,  # Pad to the longest sequence in the batch
            truncation=True,  # Ensure we don't exceed model's max length
        )
        return {k: v.to(device) for k, v in inputs.items()}


# ----------------------------
# Generation helpers
# ----------------------------
@torch.inference_mode()
def generate_once(model, model_inputs, max_new_tokens: int):
    # eos_token_id=-1 prevents early stop; disable_compile=True matches baseline
    return model.generate(
        **model_inputs,
        do_sample=False,
        temperature=None,
        max_new_tokens=max_new_tokens,
        eos_token_id=-1,
        disable_compile=True,
        return_dict_in_generate=True,  # so we can count actual generated length
    )


def run_memory_probe(model, model_inputs, device, max_new_tokens: int):
    torch.cuda.reset_peak_memory_stats(device)
    clear_cuda()

    # One measured pass
    torch.cuda.synchronize(device)
    _out = generate_once(model, model_inputs, max_new_tokens)
    torch.cuda.synchronize(device)

    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return to_gib(peak_alloc), to_gib(peak_reserved)


def measure_latency_with_torch_benchmark(model, model_inputs, max_new_tokens: int):
    # Closure for timing; torch.utils.benchmark will handle CUDA syncs.
    def gen_fn():
        generate_once(model, model_inputs, max_new_tokens)

    t = benchmark.Timer(
        stmt="gen_fn()",
        globals={"gen_fn": gen_fn},
        num_threads=NUM_THREADS,
    )
    m = t.blocked_autorange(min_run_time=TIMER_MIN_RUNTIME_S)
    return float(m.median)  # seconds


# ----------------------------
# Benchmark driver
# ----------------------------
def benchmark_scenario(attn_implementation: str, batch_size: int = 1) -> ScenarioResult:
    set_determinism()
    tokenizer, model, device = load_model(attn_implementation)
    inputs = build_inputs(tokenizer, device, batch_size)

    # Warmups (not measured) - use smaller token budget for larger batches to avoid OOM
    # Scale down the warmup tokens based on batch size to prevent memory issues
    if batch_size >= 32:
        warmup_tokens = min(256, min(TOKEN_BUDGETS) // 2)  # Very conservative for large batches
    elif batch_size >= 16:
        warmup_tokens = min(TOKEN_BUDGETS) // 2  # Half the smallest budget for medium-large batches
    elif batch_size > 4:
        warmup_tokens = min(TOKEN_BUDGETS)  # Smallest budget for moderately large batches
    else:
        warmup_tokens = TOKEN_BUDGETS[0]  # First budget for small batches
    for _ in range(WARMUP_GENERATIONS):
        _ = generate_once(model, inputs, max_new_tokens=warmup_tokens)
        torch.cuda.synchronize(device)

    result = ScenarioResult(
        label=f"attn_implementation={attn_implementation}",
        batch_size=batch_size
    )

    for toks in TOKEN_BUDGETS:
        try:
            # Clear memory before each token budget test
            clear_cuda()
            
            # Timing via torch.benchmark
            latency_s = measure_latency_with_torch_benchmark(model, inputs, toks)

            # Actual generated token count (sanity; should equal toks per item in batch)
            out = generate_once(model, inputs, toks)
            actual_tokens_per_item = (out.sequences.shape[1] - inputs["input_ids"].shape[1])
            total_tokens = actual_tokens_per_item * batch_size
            del out
            torch.cuda.synchronize(device)

            # Memory: take median of multiple probes
            allocs, reserveds = [], []
            for _ in range(MEM_REPEATS):
                pa, pr = run_memory_probe(model, inputs, device, toks)
                allocs.append(pa)
                reserveds.append(pr)

            med_alloc = statistics.median(allocs)
            med_reserved = statistics.median(reserveds)

            result.by_tokens[toks] = Point(
                latency_s=latency_s,
                tokens=total_tokens,
                toks_per_s=(total_tokens / latency_s) if latency_s > 0 else float("nan"),
                peak_alloc_gib=med_alloc,
                peak_reserved_gib=med_reserved,
                batch_size=batch_size,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM for batch_size={batch_size}, tokens={toks}: {e}")
            # Clear memory and continue with next token budget
            clear_cuda()
            continue
        except Exception as e:
            print(f"Error for batch_size={batch_size}, tokens={toks}: {e}")
            clear_cuda()
            continue

    # Cleanup
    del tokenizer, model, inputs
    clear_cuda()
    return result


def print_comparison(results):
    # Pretty table per token budget
    print("\n=== Throughput & Memory Comparison (Single H100) ===")
    print(f"Model: {MODEL_ID}")
    print(f"Token budgets: {TOKEN_BUDGETS}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"torch.utils.benchmark min_run_time={TIMER_MIN_RUNTIME_S}s | mem_repeats={MEM_REPEATS}")
    print("-" * 140)

    header = (
        f"{'Attention Implementation':<40} {'Batch':>6} {'Tokens':>8} "
        f"{'Median Latency (s)':>20} {'Tokens/s':>12} "
        f"{'Peak Alloc (GiB)':>18} {'Peak Reserved (GiB)':>20}"
    )
    print(header)
    print("-" * 140)

    for toks in TOKEN_BUDGETS:
        for res in results:
            if toks in res.by_tokens:
                p = res.by_tokens[toks]
                print(
                    f"{res.label:<40} {res.batch_size:>6d} {toks:>8d} "
                    f"{p.latency_s:>20.3f} {p.toks_per_s:>12.1f} "
                    f"{p.peak_alloc_gib:>18.2f} {p.peak_reserved_gib:>20.2f}"
                )
            else:
                print(
                    f"{res.label:<40} {res.batch_size:>6d} {toks:>8d} "
                    f"{'OOM/FAILED':>20} {'N/A':>12} "
                    f"{'N/A':>18} {'N/A':>20}"
                )
        print("-" * 140)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Optional: ensure only one visible GPU (uncomment if needed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    assert torch.cuda.is_available(), "CUDA is required"
    assert torch.device(DEVICE).type == "cuda"
    torch.set_num_threads(NUM_THREADS)

    results = []
    for attn_impl in ATTN_IMPLEMENTATIONS:
        for batch_size in BATCH_SIZES:
            print(f"\n>>> Benchmarking with attn_implementation={attn_impl}, batch_size={batch_size}")
            try:
                res = benchmark_scenario(attn_implementation=attn_impl, batch_size=batch_size)
                results.append(res)
            except Exception as e:
                print(f"Failed to benchmark {attn_impl} with batch_size={batch_size}: {e}")
                continue

    if results:
        print_comparison(results)
    else:
        print("No successful benchmarks completed.")