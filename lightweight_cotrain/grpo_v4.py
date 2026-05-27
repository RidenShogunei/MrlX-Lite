"""
GRPO v4 最终版 - 全部50题 + 低lr + 奖励阈值过滤
"""
import os, sys, re, torch
from typing import List
from pathlib import Path
from dataclasses import dataclass
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from math_environment import MathEnvironment, MathReward, MathTask

_builtin_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _builtin_print(*args, **kwargs)

@dataclass
class CoTrainConfig:
    base_model: str = "./models/qwen/Qwen2___5-1___5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lr: float = 5e-6  # 保守学习率
    group_size: int = 4
    batch_size: int = 4
    max_subtasks: int = 3
    max_response_len: int = 128
    sync_interval: int = 5
    reward_threshold: float = 0.6  # 只训练高奖励样本
    save_dir: str = "./sft_grpo_v4"
    device: str = "cuda:0"


class SharedModel:
    MAIN_ADAPTER = "main"
    SUB_ADAPTER = "sub"

    def __init__(self, model_path: str, config: CoTrainConfig):
        self.config = config
        self.device = config.device

        print(f"\n[系统] 加载基础模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, device_map={"": self.device}, low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

        lora_config = LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config, adapter_name=self.MAIN_ADAPTER)
        self.model.add_adapter(self.SUB_ADAPTER, lora_config)
        self.model.print_trainable_parameters()

        self.optimizers = {
            self.MAIN_ADAPTER: torch.optim.AdamW(self.model.parameters(), lr=config.lr),
            self.SUB_ADAPTER: torch.optim.AdamW(self.model.parameters(), lr=config.lr),
        }
        print(f"[系统] 显存: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def load_sft_weights(self):
        print("\n[系统] 加载 SFT 权重...")
        for name, folder in [("main","main_agent"),("sub","sub_agent")]:
            self.model.set_adapter(name)
            sd = load_file(f"./sft_checkpoints/{folder}/{name}/adapter_model.safetensors", device="cpu")
            fixed = {}
            old_suf = f".{name}.weight"
            for k,v in sd.items():
                if old_suf in k:
                    fixed[k.replace(old_suf, ".default.weight")] = v
            self.model.load_state_dict(fixed, strict=False)
            print(f"   ✅ {name} agent")
        print("   ✅ SFT 权重加载完成！")

    def set_adapter(self, name: str):
        if name != self.model.active_adapters[0]:
            self.model.set_adapter(name)

    def generate_one(self, adapter_name: str, prompt: str, max_tokens: int = None) -> str:
        self.set_adapter(adapter_name)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens or self.config.max_response_len,
                temperature=0.8, top_p=0.95, do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        plen = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()

    def sft_step(self, adapter_name: str, prompt: str, response: str, weight: float = 1.0):
        self.set_adapter(adapter_name)
        full = prompt + response
        enc = self.tokenizer(full, return_tensors="pt", truncation=True, max_length=1024)
        enc = {k: v.to(self.device) for k,v in enc.items()}
        prompt_toks = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        plen = len(prompt_toks)

        labels = enc["input_ids"].clone()
        labels[0, :plen] = -100

        out = self.model(**enc)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        weighted_loss = loss * max(weight, 0.1)

        self.optimizers[adapter_name].zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizers[adapter_name].step()
        return loss.item()

    def save_lora(self, adapter_name: str, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path, adapter_name=adapter_name)
        self.tokenizer.save_pretrained(save_path)


class GRPOTrainerV4:
    def __init__(self, config: CoTrainConfig, env=None):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.env = env or MathEnvironment()
        self.model = None

    def build_main_prompt(self, task: MathTask) -> str:
        msg = [
            {"role":"system","content":(
                "你是数学解题器，按格式回答：\n"
                "<thinking>思考过程</thinking>\n"
                "[tool_call]计算内容[/tool_call]\n"
                "<result>数字答案</result>"
            )},
            {"role":"user","content": f"问题: {task.question}"}
        ]
        return self.model.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def build_sub_prompt(self, subtask: str) -> str:
        msg = [
            {"role":"system","content":"执行计算，格式：<thinking>过程</thinking><result>数字</result>"},
            {"role":"user","content": f"计算: {subtask}"}
        ]
        return self.model.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def parse_tool_calls(self, response: str) -> List[str]:
        sts = []
        for pat in [
            r'\[tool_call\]\s*(.+?)\s*\[/tool_call\]',
            r'<tool_call[^>]*>\s*(.+?)\s*</tool_call>',
        ]:
            for m in re.findall(pat, response, re.DOTALL):
                s = re.sub(r'<[^>]+>', '', m.strip()).strip()
                if 3 < len(s) < 200: sts.append(s)
        if not sts:
            for line in response.split('\n'):
                for kw in ['计算','calculate','compute','solve','×','+','-','÷']:
                    idx = line.lower().find(kw)
                    if idx >= 0:
                        s = line[idx:].strip()[:200]
                        s = re.sub(r'<[^>]+>', '', s).strip()
                        if 3 < len(s) < 200: sts.append(s); break
        return sts[:self.config.max_subtasks]

    def parse_result(self, response: str) -> str:
        m = re.search(r'<result>\s*(-?\d+\.?\d*)\s*</result>', response)
        if m: return m.group(1)
        nums = re.findall(r'-?\d+\.?\d*', response)
        return nums[-1] if nums else ""

    def run_episode(self, task: MathTask):
        prompt_m = self.build_main_prompt(task)
        candidates = []
        for _ in range(self.config.group_size):
            resp = self.model.generate_one(SharedModel.MAIN_ADAPTER, prompt_m)
            reward = MathReward.compute_main_reward(task, resp, [])
            candidates.append((resp, reward))
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_resp, best_rew = candidates[0]

        sub_candidates = []
        subtasks = self.parse_tool_calls(best_resp)
        for st in subtasks:
            prompt_s = self.build_sub_prompt(st)
            for _ in range(2):
                sr = self.model.generate_one(SharedModel.SUB_ADAPTER, prompt_s, max_tokens=80)
                sr_rew = MathReward.compute_sub_reward(st, sr, main_score=best_rew, task_answer=task.answer)
                sub_candidates.append((prompt_s, sr, sr_rew))
        sub_candidates.sort(key=lambda x: x[2], reverse=True)

        return candidates, sub_candidates

    def train(self, tasks: List[MathTask], num_iterations: int = 20):
        print(f"[系统] Best-of-N+SFT 训练，{num_iterations}轮，全部{len(tasks)}题")
        print(f"[系统] lr={self.config.lr}, group={self.config.group_size}, threshold={self.config.reward_threshold}")

        self.model = SharedModel(self.config.base_model, self.config)
        self.model.load_sft_weights()
        self.model.model.train()

        for it in range(num_iterations):
            print(f"\n===== Iter {it+1}/{num_iterations} =====")
            main_rews, sub_rews = [], []
            main_updates, sub_updates = 0, 0

            for task in tasks:
                main_cands, sub_cands = self.run_episode(task)
                best_resp, best_rew = main_cands[0]
                main_rews.append(best_rew)

                # 仅训练高质量样本
                if best_rew > self.config.reward_threshold:
                    self.model.sft_step(SharedModel.MAIN_ADAPTER,
                        self.build_main_prompt(task), best_resp, weight=best_rew)
                    main_updates += 1

                for sp, sr, sr_rew in sub_cands[:2]:
                    if sr_rew > self.config.reward_threshold:
                        self.model.sft_step(SharedModel.SUB_ADAPTER, sp, sr, weight=sr_rew)
                        sub_updates += 1
                    sub_rews.append(sr_rew)

            avg_mr = sum(main_rews)/len(main_rews)
            avg_sr = sum(sub_rews)/max(len(sub_rews),1)
            print(f"  Main R={avg_mr:.3f} [updates={main_updates}/{len(tasks)}] | "
                  f"Sub R={avg_sr:.3f} [updates={sub_updates}]")

            if (it+1) % self.config.sync_interval == 0:
                print(f"  [保存] step={it+1}")
                for name in ["main","sub"]:
                    self.model.save_lora(name, str(self.save_dir / f"{name}_step_{it+1}"))

        print(f"\n✅ 训练完成！")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    config = CoTrainConfig(lr=5e-6, group_size=4, reward_threshold=0.6, sync_interval=5, save_dir="./sft_grpo_v4")
    env = MathEnvironment(seed=42)
    trainer = GRPOTrainerV4(config, env=env)
    trainer.train(env.tasks, num_iterations=20)
