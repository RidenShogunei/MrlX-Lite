"""
SFT 监督微调脚本
使用高质量格式数据对 Main Agent 和 Sub Agent 进行 LoRA 微调

训练策略：
1. 先分别训练 Main 和 Sub 的格式输出能力
2. 使用较低学习率，避免破坏预训练知识
3. 训练少量 epoch，防止过拟合
"""

import os
import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class CoTrainConfig:
    """协同训练配置"""
    base_model: str = "./models/qwen/Qwen2___5-1___5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    lr: float = 3e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    save_dir: str = "./sft_checkpoints"
    device: str = "cuda:0"
    use_4bit: bool = False
    max_length: int = 1024


def load_sft_data(data_path: str) -> tuple:
    """加载 SFT 数据，按类别分组"""
    main_samples = []
    sub_samples = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item['category'] == 'main':
                main_samples.append(item)
            else:
                sub_samples.append(item)

    return main_samples, sub_samples


def _format_messages(messages: List[Dict]) -> str:
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"[{role}]\n{content}\n"
    return text


def prepare_training_data(samples: List[Dict], tokenizer, max_length: int = 512) -> List[Dict]:
    """准备训练数据"""
    encodings = []

    for sample in samples:
        messages = sample['messages']
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("SFT sample must end with an assistant message")

        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        text = prompt_text + messages[-1]["content"] + (tokenizer.eos_token or "")

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        prompt_encoding = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        prompt_len = min(prompt_encoding['input_ids'].shape[-1], labels.shape[0])
        labels[:prompt_len] = -100
        if (labels != -100).sum().item() == 0:
            continue

        encodings.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        })

    return encodings


def save_selected_adapter(model, tokenizer, output_dir: str, adapter_name: str):
    """Save only one LoRA adapter, while supporting multiple PEFT versions."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        model.save_pretrained(output_dir, selected_adapters=[adapter_name])
    except TypeError:
        model.save_pretrained(output_dir, adapter_name=adapter_name)
    tokenizer.save_pretrained(output_dir)


def train_lora(
    model,
    tokenizer,
    train_data: List[Dict],
    config: CoTrainConfig,
    adapter_name: str,
    output_dir: str
):
    """训练单个 LoRA 适配器"""
    print(f"\n{'='*60}")
    print(f"开始训练 {adapter_name} Agent LoRA...")
    print(f"{'='*60}")
    print(f"训练样本数: {len(train_data)}")
    print(f"Epochs: {config.num_epochs}")
    print(f"学习率: {config.lr}")

    adapter_id = adapter_name.lower()
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    model.train()

    for epoch in range(config.num_epochs):
        total_loss = 0
        num_batches = 0

        # 打乱数据
        indices = torch.randperm(len(train_data)).tolist()

        for i in range(0, len(train_data), config.batch_size):
            batch_indices = indices[i:i + config.batch_size]
            if len(batch_indices) < config.batch_size:
                continue

            # 准备 batch
            input_ids_list = [train_data[idx]['input_ids'] for idx in batch_indices]
            attention_mask_list = [train_data[idx]['attention_mask'] for idx in batch_indices]
            label_ids_list = [train_data[idx]['labels'] for idx in batch_indices]

            # Padding
            max_len = max(ids.shape[0] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_mask = []
            labels_list = []

            for ids, mask, labels in zip(input_ids_list, attention_mask_list, label_ids_list):
                pad_len = max_len - ids.shape[0]
                padded_ids = torch.cat([
                    ids,
                    torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
                ])
                padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
                padded_labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

                padded_input_ids.append(padded_ids)
                padded_attention_mask.append(padded_mask)
                labels_list.append(padded_labels)

            # Stack
            input_ids = torch.stack(padded_input_ids).to(config.device)
            attention_mask = torch.stack(padded_attention_mask).to(config.device)
            labels = torch.stack(labels_list).to(config.device)

            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # 计算 loss（只对 labels != -100 的 token）
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if (shift_labels != -100).sum().item() == 0:
                continue

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            if not torch.isfinite(loss):
                continue

            # Backward
            loss.backward()
            total_loss += loss.item()
            num_batches += 1

            if num_batches % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if num_batches % config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{config.num_epochs} - Loss: {avg_loss:.4f}")

    # 保存 LoRA 权重
    save_selected_adapter(model, tokenizer, output_dir, adapter_id)
    print(f"LoRA 权重已保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Main/Sub LoRA adapters from SFT JSONL data.")
    parser.add_argument("--data-path", default=str(Path(__file__).parent / "sft_data.jsonl"))
    parser.add_argument("--save-dir", default="./sft_checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-model", default="./models/qwen/Qwen2___5-1___5B-Instruct")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    config = CoTrainConfig(
        base_model=args.base_model,
        save_dir=args.save_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        max_length=args.max_length,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 加载 tokenizer
    print(f"\n[系统] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    print(f"[系统] 加载基础模型...")
    quantization_config = None
    if config.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map={"": config.device},
        low_cpu_mem_usage=True,
    )

    # 加载 SFT 数据
    main_samples, sub_samples = load_sft_data(args.data_path)

    print(f"[系统] 加载 SFT 数据:")
    print(f"  Main Agent: {len(main_samples)} 条")
    print(f"  Sub Agent: {len(sub_samples)} 条")

    # 准备训练数据
    print(f"[系统] 准备训练数据...")
    main_train_data = prepare_training_data(main_samples, tokenizer, max_length=config.max_length)
    sub_train_data = prepare_training_data(sub_samples, tokenizer, max_length=config.max_length)

    # 创建 LoRA 配置
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 训练 Main Agent
    print(f"\n[系统] 添加 Main Agent LoRA 适配器...")
    model = get_peft_model(model, lora_config, adapter_name="main")
    model.print_trainable_parameters()

    train_lora(
        model,
        tokenizer,
        main_train_data,
        config,
        "Main",
        os.path.join(config.save_dir, "main_agent")
    )

    # 训练 Sub Agent
    print(f"\n[系统] 添加 Sub Agent LoRA 适配器...")
    model.add_adapter("sub", lora_config)

    train_lora(
        model,
        tokenizer,
        sub_train_data,
        config,
        "Sub",
        os.path.join(config.save_dir, "sub_agent")
    )

    print(f"\n[系统] SFT 训练完成！")
    print(f"[系统] 权重保存至: {config.save_dir}")


if __name__ == "__main__":
    main()
