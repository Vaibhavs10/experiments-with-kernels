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