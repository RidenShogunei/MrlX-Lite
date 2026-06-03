"""
GRPO v4 最终版 - 全部50题 + 低lr + 奖励阈值过滤
"""
import argparse
import os, sys, re, torch
import shutil
from typing import List, Optional
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
    reward_mode: str = "hybrid"  # "hybrid" for warmup, "strict" for MrlX binary reward
    canonicalize_outputs: bool = True
    save_dir: str = "./sft_grpo_v4"
    sft_dir: str = "./sft_checkpoints"
    main_lora_path: Optional[str] = None
    sub_lora_path: Optional[str] = None
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
        explicit_paths = {
            "main": self.config.main_lora_path,
            "sub": self.config.sub_lora_path,
        }
        for name, folder in [("main","main_agent"),("sub","sub_agent")]:
            adapter_dir = Path(explicit_paths[name]) if explicit_paths[name] else Path(self.config.sft_dir) / folder / name
            adapter_path = adapter_dir / "adapter_model.safetensors"
            if not adapter_path.exists():
                raise FileNotFoundError(
                    f"SFT adapter not found: {adapter_path}. "
                    "Run sft_trainer.py first or set CoTrainConfig.sft_dir/main_lora_path/sub_lora_path."
                )
            self.model.set_adapter(name)
            sd = load_file(str(adapter_path), device="cpu")
            fixed = {}
            for k,v in sd.items():
                if f".{name}.weight" in k:
                    fixed[k] = v
                elif ".default.weight" in k:
                    fixed[k.replace(".default.weight", f".{name}.weight")] = v
                elif ".lora_A.weight" in k:
                    fixed[k.replace(".lora_A.weight", f".lora_A.{name}.weight")] = v
                elif ".lora_B.weight" in k:
                    fixed[k.replace(".lora_B.weight", f".lora_B.{name}.weight")] = v
            if not fixed:
                raise RuntimeError(f"No LoRA weights mapped from {adapter_path}")
            self.model.load_state_dict(fixed, strict=False)
            print(f"   [OK] {name} agent")
        print("   [OK] SFT 权重加载完成！")

    def set_adapter(self, name: str):
        if name != self.model.active_adapters[0]:
            self.model.set_adapter(name)

    def generate_one(self, adapter_name: str, prompt: str, max_tokens: int = None,
                     response_prefix: str = "", canonicalizer=None) -> str:
        self.set_adapter(adapter_name)
        full_prompt = prompt + response_prefix
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens or self.config.max_response_len,
                temperature=0.6, top_p=0.9, do_sample=True,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        plen = inputs["input_ids"].shape[1]
        generated = self.tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
        response = MathReward.truncate_at_first_result(response_prefix + generated).strip()
        if self.config.canonicalize_outputs and canonicalizer is not None:
            response = canonicalizer(response)
        return response

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
        try:
            self.model.save_pretrained(save_path, selected_adapters=[adapter_name])
        except TypeError:
            self.model.save_pretrained(save_path, adapter_name=adapter_name)
        nested = Path(save_path) / adapter_name
        if nested.is_dir() and (nested / "adapter_config.json").exists():
            for item in nested.iterdir():
                target = Path(save_path) / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            nested.rmdir()
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
                "<result>数字答案</result>\n"
                "只输出一段结果，写完 </result> 后立刻停止。"
            )},
            {"role":"user","content": f"问题: {task.question}"}
        ]
        return self.model.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def build_sub_prompt(self, subtask: str) -> str:
        msg = [
            {"role":"system","content":"执行计算，格式：<thinking>过程</thinking><result>数字</result>。写完 </result> 后立刻停止。"},
            {"role":"user","content": f"计算: {subtask}"}
        ]
        return self.model.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def compute_main_reward(self, task: MathTask, response: str) -> float:
        if self.config.reward_mode == "strict":
            return MathReward.compute_main_reward(task, response, [])
        if self.config.reward_mode == "hybrid":
            return MathReward.compute_main_reward_hybrid(task, response, [])
        raise ValueError(f"Unknown reward_mode: {self.config.reward_mode}")

    def evaluate_tasks(self, tasks: List[MathTask], n_samples: int = 2) -> dict:
        """Lightweight validation with the same constrained/canonical generation path."""
        if not tasks:
            return {"strict": 0.0, "hybrid": 0.0, "correct": 0.0, "best_strict": 0.0}

        strict_scores, hybrid_scores, correct_flags = [], [], []
        best_strict_total = 0.0
        self.model.model.eval()
        for task in tasks:
            prompt = self.build_main_prompt(task)
            task_strict = []
            for _ in range(n_samples):
                resp = self.model.generate_one(
                    SharedModel.MAIN_ADAPTER,
                    prompt,
                    response_prefix="<thinking>",
                    canonicalizer=MathReward.canonicalize_main_response,
                )
                strict = MathReward.compute_main_reward(task, resp, [])
                hybrid = MathReward.compute_main_reward_hybrid(task, resp, [])
                pred = MathEnvironment.extract_number(resp)
                strict_scores.append(strict)
                hybrid_scores.append(hybrid)
                correct_flags.append(1.0 if MathEnvironment.check_answer(pred, task.answer) else 0.0)
                task_strict.append(strict)
            best_strict_total += max(task_strict)
        self.model.model.train()
        total = max(len(strict_scores), 1)
        return {
            "strict": sum(strict_scores) / total,
            "hybrid": sum(hybrid_scores) / total,
            "correct": sum(correct_flags) / total,
            "best_strict": best_strict_total / len(tasks),
        }

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
            resp = self.model.generate_one(
                SharedModel.MAIN_ADAPTER,
                prompt_m,
                response_prefix="<thinking>",
                canonicalizer=MathReward.canonicalize_main_response,
            )
            reward = self.compute_main_reward(task, resp)
            strict_reward = MathReward.compute_main_reward(task, resp, [])
            hybrid_reward = MathReward.compute_main_reward_hybrid(task, resp, [])
            candidates.append((resp, reward, strict_reward, hybrid_reward))
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_resp, best_rew, _, _ = candidates[0]

        sub_candidates = []
        subtasks = self.parse_tool_calls(best_resp)
        for st in subtasks:
            prompt_s = self.build_sub_prompt(st)
            for _ in range(2):
                sr = self.model.generate_one(
                    SharedModel.SUB_ADAPTER,
                    prompt_s,
                    max_tokens=80,
                    response_prefix="<thinking>",
                    canonicalizer=MathReward.canonicalize_sub_response,
                )
                sr_rew = MathReward.compute_sub_reward(st, sr, main_score=best_rew, task_answer=task.answer)
                sub_candidates.append((prompt_s, sr, sr_rew))
        sub_candidates.sort(key=lambda x: x[2], reverse=True)

        return candidates, sub_candidates

    def train(self, tasks: List[MathTask], num_iterations: int = 20,
              val_tasks: List[MathTask] = None, eval_samples: int = 2):
        print(f"[系统] Best-of-N+SFT 训练，{num_iterations}轮，全部{len(tasks)}题")
        print(f"[系统] lr={self.config.lr}, group={self.config.group_size}, "
              f"threshold={self.config.reward_threshold}, reward={self.config.reward_mode}")

        self.model = SharedModel(self.config.base_model, self.config)
        self.model.load_sft_weights()
        self.model.model.train()
        best_val = -1.0

        for it in range(num_iterations):
            print(f"\n===== Iter {it+1}/{num_iterations} =====")
            main_rews, main_strict_rews, main_hybrid_rews, sub_rews = [], [], [], []
            main_updates, sub_updates = 0, 0

            for task in tasks:
                main_cands, sub_cands = self.run_episode(task)
                best_resp, best_rew, best_strict, best_hybrid = main_cands[0]
                main_rews.append(best_rew)
                main_strict_rews.append(best_strict)
                main_hybrid_rews.append(best_hybrid)

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
            avg_strict = sum(main_strict_rews)/len(main_strict_rews)
            avg_hybrid = sum(main_hybrid_rews)/len(main_hybrid_rews)
            avg_sr = sum(sub_rews)/max(len(sub_rews),1)
            print(f"  Main R={avg_mr:.3f} [updates={main_updates}/{len(tasks)}] | "
                  f"strict={avg_strict:.3f} hybrid={avg_hybrid:.3f} | "
                  f"Sub R={avg_sr:.3f} [updates={sub_updates}]")

            if val_tasks:
                val = self.evaluate_tasks(val_tasks, n_samples=eval_samples)
                print(f"  [val] strict={val['strict']:.3f} best={val['best_strict']:.3f} "
                      f"hybrid={val['hybrid']:.3f} correct={val['correct']:.3f}")
                if val["best_strict"] > best_val:
                    best_val = val["best_strict"]
                    print(f"  [best] 保存验证集最佳 checkpoint (best_strict={best_val:.3f})")
                    for name in ["main", "sub"]:
                        self.model.save_lora(name, str(self.save_dir / "best"))

            if (it+1) % self.config.sync_interval == 0:
                print(f"  [保存] step={it+1}")
                for name in ["main","sub"]:
                    self.model.save_lora(name, str(self.save_dir / f"{name}_step_{it+1}"))

        print(f"\n[OK] 训练完成！")


def parse_args():
    parser = argparse.ArgumentParser(description="Run lightweight GRPO warmup/training.")
    parser.add_argument("--tasks", type=int, default=50, help="Number of math tasks to train on.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of training iterations.")
    parser.add_argument("--group-size", type=int, default=4, help="Best-of-N samples per task.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--reward-threshold", type=float, default=0.3, help="Minimum reward to train a sample.")
    parser.add_argument("--reward-mode", choices=["hybrid", "strict"], default="hybrid")
    parser.add_argument("--sync-interval", type=int, default=5, help="Checkpoint save interval.")
    parser.add_argument("--max-response-len", type=int, default=128)
    parser.add_argument("--save-dir", default="./sft_grpo_v4")
    parser.add_argument("--sft-dir", default="./sft_checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=99)
    parser.add_argument("--val-tasks", type=int, default=0, help="Number of held-out validation tasks.")
    parser.add_argument("--eval-samples", type=int, default=2)
    parser.add_argument("--no-canonicalize", action="store_true", help="Disable output canonicalization.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    config = CoTrainConfig(
        lr=args.lr,
        group_size=args.group_size,
        reward_threshold=args.reward_threshold,
        reward_mode=args.reward_mode,
        canonicalize_outputs=not args.no_canonicalize,
        sync_interval=args.sync_interval,
        max_response_len=args.max_response_len,
        save_dir=args.save_dir,
        sft_dir=args.sft_dir,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    env = MathEnvironment(seed=args.seed)
    val_tasks = MathEnvironment(seed=args.val_seed).tasks[-args.val_tasks:] if args.val_tasks > 0 else None
    trainer = GRPOTrainerV4(config, env=env)
    trainer.train(env.tasks[:args.tasks], num_iterations=args.iterations,
                  val_tasks=val_tasks, eval_samples=args.eval_samples)
