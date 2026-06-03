"""Quick verification that Qwen3.5-9B loads correctly from cache."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

BASE_MODEL = "/home/jinxu/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    print("Loading model to GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    print(f"Model loaded: {model.config.model_type}")
    print(f"Model device: {next(model.parameters()).device}")

    print("Running a simple forward pass...")
    inputs = tokenizer("What is the capital of France?", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generation result: {result}")
    print("\n✅ Verification passed!")

if __name__ == "__main__":
    main()
