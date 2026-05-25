"""
轻量级多智能体协同训练系统 - Main Agent → Sub Agent (任务分解) 模式
基于 MrlX-DeepResearch 架构简化版

核心优化：
1. 单 base 模型 + 多 LoRA 适配器（显存省一半）
2. 每轮只加载一次（rollout + train 合并）
3. LoRA 切换通过 PEFT add_adapter/set_adapter 实现

Main Agent (探索者): 接收复杂任务，分解为子任务，输出结构化格式
Sub Agent (执行者): 接收子任务，执行并返回结果

基于 Transformers + PEFT(LoRA) + 自定义 GRPO 循环
模型：Qwen2.5-0.5B (单卡 8GB 可跑)
"""

import os
import json
import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import gc
from collections import deque

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

from math_environment import MathEnvironment, MathReward, MathTask


@dataclass
class CoTrainConfig:
    """协同训练配置"""
    base_model: str = "./models/qwen/Qwen2___5-0___5B-Instruct"

    # LoRA 配置
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # 训练配置
    lr: float = 1e-4
    batch_size: int = 2
    group_size: int = 4
    max_subtasks: int = 3
    max_response_len: int = 2048

    # GRPO 配置
    eps_clip: float = 0.2
    kl_coef: float = 0.05

    # 训练频率
    sync_interval: int = 50

    # 路径
    save_dir: str = "./cotrain_checkpoints_math_improved"

    # 设备
    device: str = "cuda:0"

    # 量化配置
    use_4bit: bool = False


class DatabaseQueue:
    """模拟跨 Agent 通信的队列"""

    def __init__(self, max_size: int = 1000):
        self.queue = deque(maxlen=max_size)
        self.completed_tasks = {}

    def commit(self, task_id: str, subtask: str, main_score: float = 0.0) -> bool:
        self.queue.append({
            "task_id": task_id,
            "subtask": subtask,
            "main_score": main_score,
            "status": "pending",
        })
        return True

    def fetch(self) -> Optional[Dict]:
        if not self.queue:
            return None
        return self.queue.popleft()

    def complete(self, task_id: str, result: str, sub_score: float = 0.0):
        self.completed_tasks[task_id] = {
            "result": result,
            "sub_score": sub_score,
        }

    def get_result(self, task_id: str) -> Optional[Dict]:
        return self.completed_tasks.get(task_id)

    def size(self) -> int:
        return len(self.queue)


class SharedModel:
    """
    共享基础模型 + 多 LoRA 适配器
    Main 和 Sub 共用一个 base 模型，通过切换 LoRA 适配器区分
    """

    MAIN_ADAPTER = "main"
    SUB_ADAPTER = "sub"

    def __init__(self, model_path: str, config: CoTrainConfig):
        self.config = config
        self.device = config.device
        self.model_path = model_path
        self.global_step_main = 0
        self.global_step_sub = 0
        self.tokenizer = None
        self.model = None
        self.optimizers = {}

        print(f"\n[系统] 加载基础模型...")
        quantization_config = None
        if config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=list(config.target_modules),
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config, adapter_name=self.MAIN_ADAPTER)
        self.model.print_trainable_parameters()

        self.optimizers[self.MAIN_ADAPTER] = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr
        )

        self.model.add_adapter(self.SUB_ADAPTER, lora_config)
        self.optimizers[self.SUB_ADAPTER] = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr
        )

        self.model.set_adapter(self.MAIN_ADAPTER)
        self.model.eval()

        print(f"[系统] 基础模型加载完成。显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def switch_adapter(self, adapter_name: str):
        """切换 LoRA 适配器"""
        if adapter_name != self.model.active_adapters[0]:
            self.model.set_adapter(adapter_name)

    def generate(self, adapter_name: str, prompts: List[str], num_return_sequences: int = 1,
                 max_new_tokens: int = None, temperature: float = 1.0) -> Tuple[List[str], List[int]]:
        """批量生成响应，返回响应和响应长度列表"""
        self.switch_adapter(adapter_name)

        responses = []
        response_lengths = []
        max_tokens = max_new_tokens or self.config.max_response_len

        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                add_special_tokens=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            prompt_tokens = inputs["input_ids"].shape[1]
            if prompt_tokens > 3500:
                print(f"[警告] Prompt 长度: {prompt_tokens} tokens (接近截断限制 4096)")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            for output in outputs:
                response = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                response_tokens = len(output) - prompt_len
                
                if response_tokens >= max_tokens - 50:
                    print(f"[警告] 响应长度: {response_tokens} tokens (接近生成长度限制 {max_tokens})")
                
                responses.append(response.strip())
                response_lengths.append(response_tokens)

        return responses, response_lengths

    def _compute_logprob_single(self, adapter_name: str, prompt: str, response: str,
                                 requires_grad: bool = False, max_seq_len: int = 2048):
        full_text = prompt + response

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)

        labels = inputs["input_ids"].clone()
        labels[0, :prompt_len] = -100

        if requires_grad:
            outputs = self.model(**inputs)
            logits = outputs.logits
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)

        mask = (shift_labels != -100).float()
        safe_labels = shift_labels.clone()
        safe_labels[safe_labels == -100] = 0

        safe_labels_expanded = safe_labels.unsqueeze(-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=safe_labels_expanded
        ).squeeze(-1)

        seq_log_prob = (token_log_probs * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)

        return seq_log_prob[0]

    def compute_logprobs(self, adapter_name: str, prompts: List[str], responses: List[str],
                         requires_grad: bool = False) -> torch.Tensor:
        self.switch_adapter(adapter_name)

        results = []
        max_seq_len = 2048 if len(responses) > 1 else 3072

        for prompt, response in zip(prompts, responses):
            lp = self._compute_logprob_single(adapter_name, prompt, response,
                                              requires_grad=requires_grad,
                                              max_seq_len=max_seq_len)
            results.append(lp)

        return torch.stack(results)

    def grpo_step(
        self,
        adapter_name: str,
        prompts: List[str],
        responses: List[List[str]],
        rewards: List[List[float]],
        old_logprobs: List[torch.Tensor],
    ) -> Dict[str, float]:
        """GRPO 训练一步"""
        self.switch_adapter(adapter_name)

        batch_size = len(prompts)
        group_size = len(responses[0]) if responses else 0

        advantages = []
        for r_group in rewards:
            r_tensor = torch.tensor(r_group, dtype=torch.float32)
            mean_r = r_tensor.mean()
            std_r = r_tensor.std(unbiased=False)
            if std_r < 1e-6 or torch.isnan(std_r):
                if len(r_group) == 1:
                    adv_group = r_group
                else:
                    adv_group = [0.0] * len(r_group)
            else:
                adv_group = ((r_tensor - mean_r) / std_r).tolist()
            advantages.append(adv_group)

        all_prompts = []
        all_responses = []
        all_advantages = []
        all_old_logprobs = []

        for i in range(batch_size):
            for j in range(group_size):
                all_prompts.append(prompts[i])
                all_responses.append(responses[i][j])
                all_advantages.append(advantages[i][j])
                all_old_logprobs.append(old_logprobs[i][j])

        batch_compute_size = 4
        new_logprobs_list = []

        for start in range(0, len(all_prompts), batch_compute_size):
            end = start + batch_compute_size
            batch_prompts = all_prompts[start:end]
            batch_responses = all_responses[start:end]

            logprobs = self.compute_logprobs(adapter_name, batch_prompts, batch_responses, requires_grad=True)
            new_logprobs_list.append(logprobs)

        new_logprobs = torch.cat(new_logprobs_list, dim=0)
        old_logprobs_tensor = torch.stack(all_old_logprobs).to(self.device)
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32).to(self.device)

        log_ratio = new_logprobs - old_logprobs_tensor
        log_ratio = torch.clamp(log_ratio, -10, 10)
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(
            ratio,
            1 - self.config.eps_clip,
            1 + self.config.eps_clip,
        ) * advantages_tensor

        policy_loss = -torch.min(surr1, surr2).mean()
        kl_penalty = log_ratio.mean()
        kl_penalty = torch.clamp(kl_penalty, -10, 10)

        loss = policy_loss + self.config.kl_coef * kl_penalty

        self.model.train()
        self.optimizers[adapter_name].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizers[adapter_name].step()
        self.model.eval()

        if adapter_name == SharedModel.MAIN_ADAPTER:
            self.global_step_main += 1
        else:
            self.global_step_sub += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "mean_reward": sum(sum(r) for r in rewards) / sum(len(r) for r in rewards),
        }

    def save_lora_weights(self, adapter_name: str, save_path: str):
        """保存指定适配器的 LoRA 权重"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path, adapter_name=adapter_name)
        self.tokenizer.save_pretrained(save_path)
        print(f"[系统] LoRA 权重已保存: {save_path} (adapter: {adapter_name})")


class CoTrainSystemOptimized:
    """
    优化后的协同训练系统
    - 单 base 模型 + 多 LoRA
    - rollout 和 train 合并，每轮只加载一次
    """

    def __init__(self, config: CoTrainConfig, env: MathEnvironment = None):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.env = env or MathEnvironment()

        self.shared_model = None
        self.db_queue = DatabaseQueue()

    def _load_model(self):
        """加载共享模型"""
        if self.shared_model is None:
            print("\n[系统] 加载共享模型...")
            self.shared_model = SharedModel(self.config.base_model, self.config)
            print(f"[系统] 模型加载完成。显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def _unload_model(self):
        """卸载模型"""
        if self.shared_model is not None:
            print("\n[系统] 卸载模型...")
            del self.shared_model
            self.shared_model = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[系统] 卸载后显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def build_main_prompt(self, task: MathTask) -> str:
        """构建 Main Agent 的 prompt"""
        system_msg = (
            "You are a math problem solver. Your goal is to break down math problems into steps.\n"
            "Use the following format:\n"
            "<thinking>Your reasoning process</thinking>\n"
            "<tool_call>subtask=\"specific calculation step\"</tool_call>\n"
            "After receiving results, provide the final answer:\n"
            "<result>Your final numerical answer</result>\n"
            "Important: The final answer must be a number."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Problem: {task.question}\nPlan and solve:"}
        ]

        prompt = self.shared_model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def build_sub_prompt(self, subtask: str) -> str:
        """构建 Sub Agent 的 prompt"""
        system_msg = (
            "You are a task executor. Your goal is to complete the given subtask.\n"
            "Use the following format:\n"
            "<thinking>Your reasoning process</thinking>\n"
            "<result>Your answer to the subtask</result>"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Subtask: {subtask}\nExecute:"}
        ]

        prompt = self.shared_model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def parse_main_response(self, response: str) -> Tuple[str, List[str]]:
        """解析 Main Agent 的响应"""
        thinking = ""
        subtasks = []

        thinking_match = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
        else:
            thinking = response[:100].strip()

        subtask_matches = re.findall(r'<tool_call>subtask="(.*?)"</tool_call>', response)
        if subtask_matches:
            subtasks = subtask_matches
        else:
            numbered = re.findall(r'(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\n\s*[-\*]\s|$)', response, re.DOTALL)
            if numbered:
                subtasks = [s.strip() for s in numbered if len(s.strip()) > 5]
            else:
                bullets = re.findall(r'(?:^|\n)\s*[-\*]\s*(.+?)(?=\n\s*[-\*]\s|$)', response, re.DOTALL)
                if bullets:
                    subtasks = [s.strip() for s in bullets if len(s.strip()) > 5]
                else:
                    if len(response.strip()) > 10:
                        subtasks = [response.strip()[:200]]

        return thinking, subtasks

    def parse_sub_response(self, response: str) -> Tuple[str, str]:
        """解析 Sub Agent 的响应"""
        thinking = ""
        result = ""

        thinking_match = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()

        result_match = re.search(r"<result>(.*?)</result>", response, re.DOTALL)
        if result_match:
            result = result_match.group(1).strip()
        else:
            result = response.strip()

        return thinking, result

    def run_episode(self, task: MathTask, group_size: int = 4) -> Tuple[List, List, List, List, List, List]:
        """运行一轮交互，返回对话和响应长度"""
        main_dialogues = []
        sub_dialogues = []
        main_rewards = []
        sub_rewards = []
        main_response_lengths = []
        sub_response_lengths = []

        main_prompts = [self.build_main_prompt(task) for _ in range(group_size)]
        main_responses, main_lengths = self.shared_model.generate(
            SharedModel.MAIN_ADAPTER, main_prompts, max_new_tokens=self.config.max_response_len
        )
        main_response_lengths.extend(main_lengths)

        for i in range(group_size):
            main_response = main_responses[i]
            thinking, subtasks = self.parse_main_response(main_response)
            subtasks = subtasks[:self.config.max_subtasks]

            if i == 0:
                print(f"\n  [Main Agent] 思考: {thinking[:200]}")
                print(f"  [Main Agent] 分解为 {len(subtasks)} 个子任务:")
                for j, st in enumerate(subtasks):
                    print(f"    {j+1}. {st[:120]}")

            sub_results = []
            sub_task_data = []
            sub_task_rewards = []

            for j, subtask in enumerate(subtasks):
                task_id = f"math_{i}_{j}"
                self.db_queue.commit(task_id, subtask, main_score=0.0)

                queue_item = self.db_queue.fetch()
                if queue_item:
                    sub_prompt = self.build_sub_prompt(queue_item["subtask"])
                    sub_resp_list, sub_lengths = self.shared_model.generate(
                        SharedModel.SUB_ADAPTER, [sub_prompt], max_new_tokens=1024
                    )
                    sub_response = sub_resp_list[0]
                    sub_response_lengths.extend(sub_lengths)

                    _, result = self.parse_sub_response(sub_response)
                    sub_results.append(result)
                    sub_task_data.append((sub_prompt, sub_response, subtask))
                    self.db_queue.complete(task_id, result, sub_score=0.0)

                    if i == 0 and j < 3:
                        print(f"    [{j+1}] Sub结果: {result[:100]}")

            main_reward = MathReward.compute_main_reward(task, main_response, sub_results)
            main_rewards.append(main_reward)
            main_dialogues.append({
                "prompt": main_prompts[i],
                "response": main_response,
                "reward": main_reward,
            })

            for sub_prompt, sub_response, subtask in sub_task_data:
                sub_reward = MathReward.compute_sub_reward(
                    subtask, sub_response, main_score=main_reward, task_answer=task.answer
                )
                sub_task_rewards.append(sub_reward)
                sub_dialogues.append({
                    "prompt": sub_prompt,
                    "response": sub_response,
                    "reward": sub_reward,
                })

            sub_rewards.extend(sub_task_rewards)

        return main_dialogues, sub_dialogues, main_rewards, sub_rewards, main_response_lengths, sub_response_lengths

    def train(self, tasks: List[MathTask], num_iterations: int = 50):
        """主训练循环 - 优化版：每轮只加载一次"""
        print(f"[系统] 开始训练，共 {num_iterations} 轮...")
        print(f"[系统] 设备: {self.config.device}")
        print(f"[系统] 优化：单模型 + 多LoRA + 每轮只加载一次")

        for iteration in range(num_iterations):
            print(f"\n{'='*50}")
            print(f"=== Iteration {iteration + 1}/{num_iterations} ===")
            print(f"{'='*50}")

            self._load_model()

            print(f"\n[系统] 收集 {len(tasks)} 个任务的 rollout 数据...")

            all_main_samples = []
            all_sub_samples = []
            all_main_lengths = []
            all_sub_lengths = []

            for task in tasks:
                main_dialogues, sub_dialogues, main_rewards, sub_rewards, main_lengths, sub_lengths = self.run_episode(
                    task, group_size=self.config.group_size
                )
                all_main_lengths.extend(main_lengths)
                all_sub_lengths.extend(sub_lengths)

                main_prompts = [d["prompt"] for d in main_dialogues]
                main_responses = [d["response"] for d in main_dialogues]
                old_lp_main = self.shared_model.compute_logprobs(
                    SharedModel.MAIN_ADAPTER, main_prompts, main_responses
                )

                for k in range(len(main_dialogues)):
                    all_main_samples.append({
                        "prompt": main_dialogues[k]["prompt"],
                        "response": main_dialogues[k]["response"],
                        "reward": main_rewards[k],
                        "old_logprob": old_lp_main[k].item(),
                    })

                for d in sub_dialogues:
                    old_lp_sub = self.shared_model.compute_logprobs(
                        SharedModel.SUB_ADAPTER, [d["prompt"]], [d["response"]]
                    )
                    all_sub_samples.append({
                        "prompt": d["prompt"],
                        "response": d["response"],
                        "reward": d["reward"],
                        "old_logprob": old_lp_sub[0].item(),
                    })

            print(f"[系统] Rollout 完成。样本: Main={len(all_main_samples)}, Sub={len(all_sub_samples)}")
            
            # 打印平均响应长度
            if all_main_lengths:
                avg_main_len = sum(all_main_lengths) / len(all_main_lengths)
                max_main_len = max(all_main_lengths)
                min_main_len = min(all_main_lengths)
                print(f"[统计] Main Agent - 平均: {avg_main_len:.1f} tokens, 最大: {max_main_len}, 最小: {min_main_len}, 限制: {self.config.max_response_len}")
            
            if all_sub_lengths:
                avg_sub_len = sum(all_sub_lengths) / len(all_sub_lengths)
                max_sub_len = max(all_sub_lengths)
                min_sub_len = min(all_sub_lengths)
                print(f"[统计] Sub Agent  - 平均: {avg_sub_len:.1f} tokens, 最大: {max_sub_len}, 最小: {min_sub_len}, 限制: 1024")

            metrics = {}

            if all_main_samples:
                for start in range(0, len(all_main_samples), self.config.batch_size):
                    batch = all_main_samples[start:start + self.config.batch_size]
                    if len(batch) < 2:
                        continue

                    prompts = [s["prompt"] for s in batch]
                    responses = [[s["response"]] for s in batch]
                    rewards = [[s["reward"]] for s in batch]
                    old_logprobs = [torch.tensor([s["old_logprob"]]) for s in batch]

                    metrics_main = self.shared_model.grpo_step(
                        SharedModel.MAIN_ADAPTER,
                        prompts,
                        responses,
                        rewards,
                        old_logprobs,
                    )
                    metrics["main"] = metrics_main

            if all_sub_samples:
                for start in range(0, len(all_sub_samples), self.config.batch_size):
                    batch = all_sub_samples[start:start + self.config.batch_size]
                    if len(batch) < 2:
                        continue

                    prompts = [s["prompt"] for s in batch]
                    responses = [[s["response"]] for s in batch]
                    rewards = [[s["reward"]] for s in batch]
                    old_logprobs = [torch.tensor([s["old_logprob"]]) for s in batch]

                    metrics_sub = self.shared_model.grpo_step(
                        SharedModel.SUB_ADAPTER,
                        prompts,
                        responses,
                        rewards,
                        old_logprobs,
                    )
                    metrics["sub"] = metrics_sub

            if metrics:
                if "main" in metrics:
                    mm = metrics["main"]
                    print(f"\n[结果] Main Agent - Loss: {mm['loss']:.4f}, "
                          f"Policy: {mm['policy_loss']:.4f}, "
                          f"KL: {mm['kl_penalty']:.4f}, "
                          f"Reward: {mm['mean_reward']:.4f}")
                if "sub" in metrics:
                    ms = metrics["sub"]
                    print(f"[结果] Sub Agent  - Loss: {ms['loss']:.4f}, "
                          f"Policy: {ms['policy_loss']:.4f}, "
                          f"KL: {ms['kl_penalty']:.4f}, "
                          f"Reward: {ms['mean_reward']:.4f}")

            if (iteration + 1) % self.config.sync_interval == 0:
                print("\n[系统] 同步权重...")
                self.shared_model.save_lora_weights(
                    SharedModel.MAIN_ADAPTER,
                    str(self.save_dir / f"lora_main_step_{iteration + 1}")
                )
                self.shared_model.save_lora_weights(
                    SharedModel.SUB_ADAPTER,
                    str(self.save_dir / f"lora_sub_step_{iteration + 1}")
                )

            self._unload_model()

        print("\n[系统] 训练完成！")


if __name__ == "__main__":
    config = CoTrainConfig(
        base_model="./models/qwen/Qwen2___5-1___5B-Instruct",
        lora_r=16,
        lora_alpha=32,
        lr=5e-5,
        batch_size=2,
        group_size=4,
        max_subtasks=2,
        max_response_len=2048,
        sync_interval=50,
        save_dir="./cotrain_checkpoints_math_15b",
        use_4bit=False,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    env = MathEnvironment(seed=42)
    tasks = env.sample_tasks(5)

    print(f"\n[系统] 数学环境已创建，共 {len(env.tasks)} 道题")
    print(f"[系统] 使用 {len(tasks)} 道题进行训练")
    for i, t in enumerate(tasks[:3]):
        print(f"  题{i+1}: {t.question} (答案: {t.answer})")

    system = CoTrainSystemOptimized(config, env=env)
    system.train(tasks, num_iterations=200)
