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
MODEL_ID = "openai/gpt-oss-20b"
DEVICE = "cuda:0"                # Single H100
SEED = 1234
NUM_THREADS = torch.get_num_threads()
TOKEN_BUDGETS = [512, 1024, 2048]
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


@dataclass
class ScenarioResult:
    label: str
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
    return tokenizer, model, model.device


def build_inputs(tokenizer, device):
    messages = [
        {"role": "system", "content": "What is Tensor Parallelism?"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)


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
def benchmark_scenario(attn_implementation: str) -> ScenarioResult:
    set_determinism()
    tokenizer, model, device = load_model(attn_implementation)
    inputs = build_inputs(tokenizer, device)

    # Warmups (not measured)
    for _ in range(WARMUP_GENERATIONS):
        _ = generate_once(model, inputs, max_new_tokens=max(TOKEN_BUDGETS))
        torch.cuda.synchronize(device)

    result = ScenarioResult(label=f"attn_implementation={attn_implementation}")

    for toks in TOKEN_BUDGETS:
        # Timing via torch.benchmark
        latency_s = measure_latency_with_torch_benchmark(model, inputs, toks)

        # Actual generated token count (sanity; should equal toks)
        out = generate_once(model, inputs, toks)
        actual_tokens = out.sequences.shape[1] - inputs["input_ids"].shape[1]
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
            tokens=actual_tokens,
            toks_per_s=(actual_tokens / latency_s) if latency_s > 0 else float("nan"),
            peak_alloc_gib=med_alloc,
            peak_reserved_gib=med_reserved,
        )

    # Cleanup
    del tokenizer, model, inputs
    clear_cuda()
    return result


def print_comparison(results):
    # Pretty table per token budget
    print("\n=== Throughput & Memory Comparison (Single H100) ===")
    print(f"Model: {MODEL_ID}")
    print(f"Token budgets: {TOKEN_BUDGETS}")
    print(f"torch.utils.benchmark min_run_time={TIMER_MIN_RUNTIME_S}s | mem_repeats={MEM_REPEATS}")
    print("-" * 120)

    header = (
        f"{'Attention Implementation':<40} {'Tokens':>8} "
        f"{'Median Latency (s)':>20} {'Tokens/s':>12} "
        f"{'Peak Alloc (GiB)':>18} {'Peak Reserved (GiB)':>20}"
    )
    print(header)
    print("-" * 120)

    for toks in TOKEN_BUDGETS:
        for res in results:
            p = res.by_tokens[toks]
            print(
                f"{res.label:<40} {toks:>8d} "
                f"{p.latency_s:>20.3f} {p.toks_per_s:>12.1f} "
                f"{p.peak_alloc_gib:>18.2f} {p.peak_reserved_gib:>20.2f}"
            )
        print("-" * 120)


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
        print(f"\n>>> Benchmarking with attn_implementation={attn_impl}")
        try:
            res = benchmark_scenario(attn_implementation=attn_impl)
            results.append(res)
        except Exception as e:
            print(f"Failed to benchmark {attn_impl}: {e}")
            continue

    if results:
        print_comparison(results)
    else:
        print("No successful benchmarks completed.")