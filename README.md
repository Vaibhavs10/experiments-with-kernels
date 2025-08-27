# Experiments with Kernels 🪛

There's a new kid in the block - Kernels to build and use faster and efficient kernels for your ML models. Kernels allows you to download and use pre-compiled compute kernels without installing and compiling them from scratch.

To put this in words, you don't need to spend 2 hours compiling flash attention in your Python Environment. You can just pull it from the hub and the kernels library will take care of pulling the right version fit to your runtime for you.

Here's how this would look like in practice to run SmolLM3 with Flash Attention. 

Installation:

```bash
uv pip install -U transformers kernels
```

Followed by:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM3-3B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="kernels-community/flash-attn",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Traditionally just installing flash attention would take up to an hour if not more leading to compute down time and not to mention how finnicky the whole process is.

Now, you can just point to the Flash Attention 3 kernel repo on the hub and that's it!

Best Part: You can swap to use Flash Attention 3 in just a line change.

```diff
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM3-3B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
-    attn_implementation="kernels-community/flash-attn",
+    attn_implementation="kernels-community/flash-attn3:flash_attention",    
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Let's run a quick benchmark:


=== Throughput & Memory Comparison (Single H100) ===  
Model: HuggingFaceTB/SmolLM3-3B  
Token budgets: [512, 2048]  
Batch sizes: [1, 16, 32]  
torch.utils.benchmark min_run_time=2.0s | mem_repeats=3

| Batch | Tokens | eager (Latency s / Tok/s / Alloc GiB / Reserved GiB) | sdpa (Latency s / Tok/s / Alloc GiB / Reserved GiB) | flash-attn3 (Latency s / Tok/s / Alloc GiB / Reserved GiB) | flash-attn (Latency s / Tok/s / Alloc GiB / Reserved GiB) |
|-------|--------|------------------------------------------------------|-----------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| 1     | 512    | 17.895 / 28.6 / 5.80 / 5.83                         | 14.180 / 36.1 / 5.80 / 5.81                        | 18.530 / 27.6 / 5.80 / 5.81                               | 18.650 / 27.5 / 5.80 / 5.81                              |
| 16    | 512    | 18.036 / 454.2 / 6.46 / 6.86                        | 17.595 / 465.6 / 6.46 / 6.86                       | 23.669 / 346.1 / 6.41 / 6.62                              | 23.612 / 346.9 / 6.41 / 6.62                             |
| 32    | 512    | 18.572 / 882.2 / 7.15 / 8.46                        | 17.478 / 937.4 / 7.15 / 8.46                       | 23.887 / 685.9 / 7.08 / 8.43                              | 24.025 / 682.0 / 7.08 / 8.47                             |
| 1     | 2048   | 71.426 / 28.7 / 5.92 / 6.04                         | 56.245 / 36.4 / 5.91 / 6.02                        | 73.922 / 27.7 / 5.91 / 6.02                               | 74.111 / 27.6 / 5.91 / 6.02                              |
| 16    | 2048   | 71.743 / 456.7 / 8.34 / 14.83                       | 70.460 / 465.1 / 8.33 / 14.83                      | 95.068 / 344.7 / 8.17 / 23.79                             | 94.579 / 346.5 / 8.17 / 23.79                            |
| 32    | 2048   | 77.889 / 841.4 / 10.91 / 39.77                      | 75.058 / 873.1 / 10.90 / 39.78                     | 96.181 / 681.4 / 10.61 / 78.71                            | 96.758 / 677.3 / 10.61 / 78.70                           |

**Legend:**  
- eager = `attn_implementation=eager`  
- sdpa = `attn_implementation=sdpa`  
- flash-attn3 = `attn_implementation=kernels-community/flash-attn3:flash_attention`  
- flash-attn = `attn_implementation=kernels-community/flash-attn`  
- Each cell: `Median Latency (s) / Tokens/s / Peak Alloc (GiB) / Peak Reserved (GiB)`
