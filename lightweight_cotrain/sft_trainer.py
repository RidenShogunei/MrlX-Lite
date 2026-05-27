"""
SFT 监督微调脚本
使用高质量格式数据对 Main Agent 和 Sub Agent 进行 LoRA 微调

训练策略：
1. 先分别训练 Main 和 Sub 的格式输出能力
2. 使用较低学习率，避免破坏预训练知识
3. 训练少量 epoch，防止过拟合
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

from math_environment import MathEnvironment


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


class SFTSFTDataset(Dataset):
    """SFT 数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample['messages']
        category = sample['category']

        # 构建对话文本
        text = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            text += f"<|{role}|>\n{content}\n"

        text += "<|end|>\n"

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # 创建 labels（只有 assistant 部分需要计算 loss）
        labels = input_ids.clone()
        role_tokens = self.tokenizer.encode("<|system|>", add_special_tokens=False)
        role_tokens += self.tokenizer.encode("<|user|>", add_special_tokens=False)

        for token_id in role_tokens:
            labels[input_ids == token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'category': category
        }


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


def prepare_training_data(samples: List[Dict], tokenizer, max_length: int = 512) -> Dict:
    """准备训练数据"""
    encodings = []

    for sample in samples:
        messages = sample['messages']
        text = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            text += f"<|{role}|>\n{content}\n"
        text += "<|end|>\n"

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        encodings.append({
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        })

    return encodings


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

            # Padding
            max_len = max(ids.shape[0] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_mask = []
            labels_list = []

            for ids, mask in zip(input_ids_list, attention_mask_list):
                pad_len = max_len - ids.shape[0]
                padded_ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
                padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])

                padded_input_ids.append(padded_ids)
                padded_attention_mask.append(padded_mask)

                # Labels：只计算 assistant 部分的 loss
                labels = padded_ids.clone()
                for token_id in [tokenizer.pad_token_id]:
                    labels[ids == token_id] = -100
                labels_list.append(labels)

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

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Backward
            loss.backward()
            total_loss += loss.item()
            num_batches += 1

            if num_batches % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{config.num_epochs} - Loss: {avg_loss:.4f}")

    # 保存 LoRA 权重
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA 权重已保存至: {output_dir}")


def main():
    config = CoTrainConfig()

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
    data_path = Path(__file__).parent / "sft_data.jsonl"
    main_samples, sub_samples = load_sft_data(str(data_path))

    print(f"[系统] 加载 SFT 数据:")
    print(f"  Main Agent: {len(main_samples)} 条")
    print(f"  Sub Agent: {len(sub_samples)} 条")

    # 准备训练数据
    print(f"[系统] 准备训练数据...")
    main_train_data = prepare_training_data(main_samples, tokenizer)
    sub_train_data = prepare_training_data(sub_samples, tokenizer)

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
