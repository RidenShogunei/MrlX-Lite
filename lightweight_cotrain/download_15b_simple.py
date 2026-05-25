from modelscope import snapshot_download
import os

print("[系统] 开始下载 Qwen2.5-1.5B-Instruct")

# 直接下载到目标位置，不移动
model_dir = snapshot_download(
    "Qwen/Qwen2.5-1.5B-Instruct",
    cache_dir="./models",
)

print(f"[系统] 下载完成: {model_dir}")

from transformers import AutoModelForCausalLM, AutoTokenizer
print("\n[系统] 验证模型...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
print(f"[系统] Tokenizer OK")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
print(f"[系统] 模型 OK，类型: {model.__class__.__name__}")
print(f"[系统] 层数: {model.config.num_hidden_layers}")
print(f"[系统] 模型配置: {model.config.model_type}")
