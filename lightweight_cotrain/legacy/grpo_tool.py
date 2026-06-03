"""GRPO-style warmup/training for the mini tool-use environment."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from generate_tool_sft_data import MAIN_SYSTEM, SUB_SYSTEM
from grpo_v4 import CoTrainConfig, SharedModel
from tool_environment import ToolEnvironment, ToolTask


@dataclass
class ToolTrainMetrics:
    reward: float
    raw_reward: float
    tool_valid: float
    updates: int
    sub_reward: float
    sub_updates: int


class ToolGRPOTrainer:
    def __init__(self, config: CoTrainConfig, env: ToolEnvironment):
        self.config = config
        self.env = env
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def build_main_prompt(self, task: ToolTask) -> str:
        user = f"products 表：\n{ToolEnvironment.render_table(task.db_rows)}\n\n问题：{task.question}"
        return self.model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": MAIN_SYSTEM},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_sub_prompt(self, tool_call_text: str) -> str:
        return self.model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SUB_SYSTEM},
                {"role": "user", "content": tool_call_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _truncate_result(text: str) -> str:
        end = text.find("</result>")
        if end >= 0:
            return text[:end + len("</result>")]
        return text

    @staticmethod
    def _extract_tool_call_block(text: str) -> str:
        match = re.search(r"\[tool_call\].*?\[/tool_call\]", text, re.DOTALL)
        return match.group(0) if match else text

    @staticmethod
    def _sub_reward(expected_output: str, response: str) -> float:
        return 1.0 if expected_output.strip() in response else 0.0

    def generate_main_candidate(self, task: ToolTask):
        prompt = self.build_main_prompt(task)
        raw = self.model.generate_one(
            SharedModel.MAIN_ADAPTER,
            prompt,
            max_tokens=self.config.max_response_len,
            response_prefix="<thinking>",
            canonicalizer=None,
        )
        fixed = ToolEnvironment.canonicalize_response(task, raw)
        ok, tool_output = ToolEnvironment.execute_tool(task, raw)
        raw_reward = ToolEnvironment.reward(task, raw)
        answer_reward = ToolEnvironment.reward(task, fixed) if ok else 0.0
        return {
            "prompt": prompt,
            "raw": raw,
            "fixed": fixed,
            "tool_valid": 1.0 if ok else 0.0,
            "tool_output": tool_output if ok else "",
            "raw_reward": raw_reward,
            "answer_reward": answer_reward,
        }

    def run_episode(self, task: ToolTask):
        candidates = [self.generate_main_candidate(task) for _ in range(self.config.group_size)]
        candidates.sort(key=lambda c: (c["answer_reward"], c["tool_valid"], c["raw_reward"]), reverse=True)
        best = candidates[0]

        sub_candidates = []
        if best["tool_valid"] > 0:
            prompt_s = self.build_sub_prompt(self._extract_tool_call_block(best["raw"]))
            target = f"<thinking>执行工具调用</thinking><result>{best['tool_output']}</result>"
            for _ in range(2):
                sub_raw = self.model.generate_one(
                    SharedModel.SUB_ADAPTER,
                    prompt_s,
                    max_tokens=120,
                    response_prefix="<thinking>",
                    canonicalizer=None,
                )
                sub_reward = self._sub_reward(best["tool_output"], sub_raw)
                sub_candidates.append({
                    "prompt": prompt_s,
                    "raw": sub_raw,
                    "target": target,
                    "reward": sub_reward,
                })
            # Always include the environment-executed target so Sub can learn from valid tools.
            sub_candidates.append({
                "prompt": prompt_s,
                "raw": target,
                "target": target,
                "reward": 1.0,
            })
            sub_candidates.sort(key=lambda c: c["reward"], reverse=True)

        return best, sub_candidates

    def evaluate(self, tasks: List[ToolTask], samples: int = 2):
        if not tasks:
            return {"tool_valid": 0.0, "raw_reward": 0.0, "answer_reward": 0.0, "best_answer_reward": 0.0}

        self.model.model.eval()
        tool_valids, raw_rewards, answer_rewards = [], [], []
        best_total = 0.0
        for task in tasks:
            task_best = 0.0
            for _ in range(samples):
                candidate = self.generate_main_candidate(task)
                tool_valids.append(candidate["tool_valid"])
                raw_rewards.append(candidate["raw_reward"])
                answer_rewards.append(candidate["answer_reward"])
                task_best = max(task_best, candidate["answer_reward"])
            best_total += task_best
        self.model.model.train()

        total = max(len(answer_rewards), 1)
        return {
            "tool_valid": sum(tool_valids) / total,
            "raw_reward": sum(raw_rewards) / total,
            "answer_reward": sum(answer_rewards) / total,
            "best_answer_reward": best_total / len(tasks),
        }

    def train(self, train_tasks: List[ToolTask], val_tasks: List[ToolTask], iterations: int, eval_samples: int):
        print(f"[tool-grpo] train={len(train_tasks)} val={len(val_tasks)} iter={iterations}")
        print(f"[tool-grpo] lr={self.config.lr} group={self.config.group_size} threshold={self.config.reward_threshold}")

        self.model = SharedModel(self.config.base_model, self.config)
        self.model.load_sft_weights()
        self.model.model.train()

        print("\n===== Tool Initial Validation =====")
        init_val = self.evaluate(val_tasks, samples=eval_samples)
        best_val = init_val["best_answer_reward"]
        print(
            f"  [val:init] answer={init_val['answer_reward']:.3f} best={init_val['best_answer_reward']:.3f} "
            f"raw={init_val['raw_reward']:.3f} tool_valid={init_val['tool_valid']:.3f}"
        )
        print(f"  [best] save initial checkpoint (best_answer={best_val:.3f})")
        for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
            self.model.save_lora(name, str(self.save_dir / "best" / name))

        for it in range(iterations):
            print(f"\n===== Tool Iter {it + 1}/{iterations} =====")
            rewards, raw_rewards, tool_valids, sub_rewards = [], [], [], []
            updates, sub_updates = 0, 0

            for task in train_tasks:
                best, sub_candidates = self.run_episode(task)
                rewards.append(best["answer_reward"])
                raw_rewards.append(best["raw_reward"])
                tool_valids.append(best["tool_valid"])

                if best["answer_reward"] >= self.config.reward_threshold:
                    self.model.sft_step(SharedModel.MAIN_ADAPTER, best["prompt"], best["fixed"], weight=best["answer_reward"])
                    updates += 1

                for sub in sub_candidates[:1]:
                    sub_rewards.append(sub["reward"])
                    if sub["reward"] >= self.config.reward_threshold:
                        self.model.sft_step(SharedModel.SUB_ADAPTER, sub["prompt"], sub["target"], weight=sub["reward"])
                        sub_updates += 1

            avg_reward = sum(rewards) / max(len(rewards), 1)
            avg_raw = sum(raw_rewards) / max(len(raw_rewards), 1)
            avg_valid = sum(tool_valids) / max(len(tool_valids), 1)
            avg_sub = sum(sub_rewards) / max(len(sub_rewards), 1)
            print(
                f"  train answer={avg_reward:.3f} raw={avg_raw:.3f} "
                f"tool_valid={avg_valid:.3f} updates={updates}/{len(train_tasks)} | "
                f"sub={avg_sub:.3f} sub_updates={sub_updates}"
            )

            val = self.evaluate(val_tasks, samples=eval_samples)
            print(
                f"  [val] answer={val['answer_reward']:.3f} best={val['best_answer_reward']:.3f} "
                f"raw={val['raw_reward']:.3f} tool_valid={val['tool_valid']:.3f}"
            )
            if val["best_answer_reward"] > best_val:
                best_val = val["best_answer_reward"]
                print(f"  [best] save best checkpoint (best_answer={best_val:.3f})")
                for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
                    self.model.save_lora(name, str(self.save_dir / "best"))

            if (it + 1) % self.config.sync_interval == 0:
                print(f"  [save] step={it + 1}")
                for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
                    self.model.save_lora(name, str(self.save_dir / f"{name}_step_{it + 1}"))

        print("\n[OK] tool GRPO complete")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Main/Sub adapters on mini tool-use tasks.")
    parser.add_argument("--base-model", default="./models/qwen/Qwen2___5-1___5B-Instruct")
    parser.add_argument("--sft-dir", default="./tool_sft_checkpoints")
    parser.add_argument("--save-dir", default="./tool_grpo_smoke")
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--val-tasks", type=int, default=20)
    parser.add_argument("--test-tasks", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=2)
    parser.add_argument("--max-response-len", type=int, default=160)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--reward-threshold", type=float, default=0.5)
    parser.add_argument("--sync-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_tasks, val_tasks, _ = ToolEnvironment.split(args.tasks, args.val_tasks, args.test_tasks, seed=args.seed)
    config = CoTrainConfig(
        base_model=args.base_model,
        lr=args.lr,
        group_size=args.group_size,
        max_response_len=args.max_response_len,
        sync_interval=args.sync_interval,
        reward_threshold=args.reward_threshold,
        save_dir=args.save_dir,
        sft_dir=args.sft_dir,
        device=device,
    )
    trainer = ToolGRPOTrainer(config, ToolEnvironment(seed=args.seed, num_tasks=args.tasks + args.val_tasks + args.test_tasks))
    trainer.train(train_tasks, val_tasks, args.iterations, args.eval_samples)


if __name__ == "__main__":
    main()
