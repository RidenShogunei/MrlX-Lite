from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer
import os

print("[系统] 开始下载 Qwen2.5-1.5B-Instruct")

save_dir = "./models/qwen/Qwen2___5-1___5B-Instruct"

if not os.path.exists(save_dir):
    print("[系统] 正在下载模型...")
    model_dir = snapshot_download(
        "Qwen/Qwen2.5-1.5B-Instruct",
        cache_dir="./models/qwen",
    )
    
    print(f"[系统] 下载路径: {model_dir}")
    print(f"[系统] 目标路径: {os.path.abspath(save_dir)}")
    
    import shutil
    
    if os.path.exists(model_dir) and not os.path.exists(save_dir):
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        shutil.move(model_dir, save_dir)
        print(f"[系统] 模型已安装到: {save_dir}")
else:
    print(f"[系统] 模型已存在: {save_dir}")

print("\n[系统] 验证模型")
try:
    tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
    print(f"[系统] Tokenizer 加载成功")
    
    model = AutoModelForCausalLM.from_pretrained(
        save_dir,
        trust_remote_code=True,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    
    print(f"[系统] 模型加载成功，层数: {model.config.num_hidden_layers}，隐藏维度: {model.config.hidden_size}")
    print(f"[系统] 参数数量: {model.config.num_hidden_layers * model.config.hidden_size / 1e9:.2f}B (估算)")
except Exception as e:
    print(f"[错误] 模型验证失败: {e}")
    import traceback
    traceback.print_exc()
