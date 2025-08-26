from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM3-3B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="kernels-community/flash-attn3:flash_attention",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))