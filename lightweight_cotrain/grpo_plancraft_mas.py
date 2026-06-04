"""Advantage-style GRPO for Plancraft Main/Sub agents."""

import argparse
import random
from pathlib import Path

import torch

from analyze_hotpotqa_mas_results import build_prompt
from analyze_plancraft_mas_results import MAIN_SYSTEM, SUB_SYSTEM, history_text
from grpo_v4 import CoTrainConfig, SharedModel
from plancraft_environment import PlancraftBenchEpisode, load_examples


def first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


class PlancraftMASGRPOTrainer:
    def __init__(
        self,
        config: CoTrainConfig,
        max_steps: int = 10,
        best_metric: str = "success_rate",
        advantage_clip: float = 2.0,
        min_advantage: float = 0.01,
        valid_weight: float = 0.2,
        step_penalty: float = 0.01,
        eval_samples: int = 1,
    ):
        self.config = config
        self.max_steps = max_steps
        self.best_metric = best_metric
        self.advantage_clip = advantage_clip
        self.min_advantage = min_advantage
        self.valid_weight = valid_weight
        self.step_penalty = step_penalty
        self.eval_samples = max(eval_samples, 1)
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def generate_action(self, adapter_name: str, prompt: str) -> str:
        text = self.model.generate_one(
            adapter_name,
            prompt,
            max_tokens=self.config.max_response_len,
            response_prefix="",
            canonicalizer=first_line,
        )
        return text

    def build_sub_prompt(self, observation: str, history) -> str:
        return build_prompt(
            self.model.tokenizer,
            SUB_SYSTEM,
            f"Current observation:\n{observation}\n\nHistory:\n{history_text(history)}",
        )

    def build_main_prompt(self, observation: str, history, sub_advice: str) -> str:
        return build_prompt(
            self.model.tokenizer,
            MAIN_SYSTEM,
            (
                f"Current observation:\n{observation}\n\n"
                f"History:\n{history_text(history)}\n\n"
                f"Sub agent advice:\n{sub_advice}"
            ),
        )

    def candidate_reward(self, result) -> float:
        valid_rate = result.valid_action_count / max(result.action_count, 1)
        return (
            (1.0 if result.success else 0.0)
            + self.valid_weight * valid_rate
            - self.step_penalty * result.steps
        )

    def generate_candidate(self, example):
        episode = PlancraftBenchEpisode(example, max_steps=self.max_steps)
        observation = episode.reset()
        history = []
        steps = []
        for _step in range(self.max_steps):
            sub_prompt = self.build_sub_prompt(observation, history)
            sub_raw = self.generate_action(SharedModel.SUB_ADAPTER, sub_prompt)
            main_prompt = self.build_main_prompt(observation, history, sub_raw)
            main_raw = self.generate_action(SharedModel.MAIN_ADAPTER, main_prompt)
            observation, _reward, terminated, truncated, _info = episode.step(main_raw)
            steps.append(
                {
                    "sub_prompt": sub_prompt,
                    "sub_raw": sub_raw,
                    "main_prompt": main_prompt,
                    "main_raw": main_raw,
                }
            )
            history.append((sub_raw, main_raw, observation))
            if terminated or truncated:
                break
        result = episode.result()
        valid_rate = result.valid_action_count / max(result.action_count, 1)
        reward = self.candidate_reward(result)
        return {
            "steps": steps,
            "result": result,
            "reward": reward,
            "success": 1.0 if result.success else 0.0,
            "valid_rate": valid_rate,
            "invalid_rate": result.invalid_action_count / max(result.action_count, 1),
            "env_reward": result.reward,
            "step_count": result.steps,
        }

    def group_advantages(self, candidates):
        values = [cand["reward"] for cand in candidates]
        mean = sum(values) / max(len(values), 1)
        var = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        std = max(var ** 0.5, 1e-6)
        for cand, value in zip(candidates, values):
            adv = (value - mean) / std
            cand["advantage"] = max(min(adv, self.advantage_clip), -self.advantage_clip)
        return candidates

    def run_group(self, example):
        candidates = [self.generate_candidate(example) for _ in range(self.config.group_size)]
        self.group_advantages(candidates)
        candidates.sort(key=lambda cand: (cand["reward"], cand["success"], cand["valid_rate"]), reverse=True)
        return candidates

    def apply_advantage_update(self, candidates):
        main_updates, sub_updates = 0, 0
        for cand in candidates:
            adv = cand["advantage"]
            if abs(adv) < self.min_advantage:
                continue
            for step in cand["steps"]:
                self.model.sft_step(SharedModel.SUB_ADAPTER, step["sub_prompt"], step["sub_raw"], weight=adv)
                self.model.sft_step(SharedModel.MAIN_ADAPTER, step["main_prompt"], step["main_raw"], weight=adv)
                sub_updates += 1
                main_updates += 1
        return main_updates, sub_updates

    @staticmethod
    def average(rows, key: str) -> float:
        return sum(row[key] for row in rows) / max(len(rows), 1)

    def evaluate(self, examples):
        self.model.model.eval()
        rows = []
        for example in examples:
            samples = [self.generate_candidate(example) for _ in range(self.eval_samples)]
            rows.extend(samples)
        best_rows = {}
        for index, _example in enumerate(examples):
            start = index * self.eval_samples
            end = start + self.eval_samples
            samples = rows[start:end]
            best_rows[index] = max(
                samples,
                key=lambda row: (row["success"], row["reward"], row["valid_rate"]),
            )
        self.model.model.train()
        return {
            "success_rate": self.average(rows, "success"),
            "best_success_rate": self.average(best_rows.values(), "success"),
            "reward": self.average(rows, "reward"),
            "best_reward": self.average(best_rows.values(), "reward"),
            "env_reward": self.average(rows, "env_reward"),
            "valid_rate": self.average(rows, "valid_rate"),
            "invalid_rate": self.average(rows, "invalid_rate"),
            "avg_steps": self.average(rows, "step_count"),
        }

    def validation_score(self, metrics):
        if self.best_metric == "success_rate":
            return metrics["success_rate"]
        if self.best_metric == "best_success_rate":
            return metrics["best_success_rate"]
        if self.best_metric == "reward":
            return metrics["reward"]
        if self.best_metric == "best_reward":
            return metrics["best_reward"]
        if self.best_metric == "valid_rate":
            return metrics["valid_rate"]
        raise ValueError(f"Unknown best_metric: {self.best_metric}")

    def save_best(self):
        for adapter_name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
            self.model.save_lora(adapter_name, str(self.save_dir / "best" / adapter_name))

    def train(self, train_examples, val_examples, iterations: int):
        print(
            f"[plancraft-mas-grpo] train={len(train_examples)} val={len(val_examples)} "
            f"iter={iterations} group={self.config.group_size} lr={self.config.lr}"
        )
        print(
            f"[plancraft-mas-grpo] max_steps={self.max_steps} valid_weight={self.valid_weight} "
            f"step_penalty={self.step_penalty} best_metric={self.best_metric}"
        )
        print(f"[plancraft-mas-grpo] eval_samples={self.eval_samples}")
        self.model = SharedModel(self.config.base_model, self.config)
        self.model.load_sft_weights()
        self.model.model.train()

        init = self.evaluate(val_examples)
        best_val = self.validation_score(init)
        print(
            f"  [val:init] success={init['success_rate']:.3f} reward={init['reward']:.3f} "
            f"best_success={init['best_success_rate']:.3f} best_reward={init['best_reward']:.3f} "
            f"env_reward={init['env_reward']:.3f} valid={init['valid_rate']:.3f} "
            f"invalid={init['invalid_rate']:.3f} steps={init['avg_steps']:.3f}"
        )
        self.save_best()

        for it in range(iterations):
            print(f"\n===== Plancraft MAS GRPO Iter {it + 1}/{iterations} =====")
            rows = []
            main_updates, sub_updates = 0, 0
            for example in train_examples:
                candidates = self.run_group(example)
                rows.append(candidates[0])
                u_main, u_sub = self.apply_advantage_update(candidates)
                main_updates += u_main
                sub_updates += u_sub
            print(
                f"  train success={self.average(rows, 'success'):.3f} "
                f"reward={self.average(rows, 'reward'):.3f} "
                f"valid={self.average(rows, 'valid_rate'):.3f} "
                f"invalid={self.average(rows, 'invalid_rate'):.3f} "
                f"steps={self.average(rows, 'step_count'):.3f} "
                f"updates main={main_updates} sub={sub_updates}"
            )
            val = self.evaluate(val_examples)
            print(
                f"  [val] success={val['success_rate']:.3f} reward={val['reward']:.3f} "
                f"best_success={val['best_success_rate']:.3f} best_reward={val['best_reward']:.3f} "
                f"env_reward={val['env_reward']:.3f} valid={val['valid_rate']:.3f} "
                f"invalid={val['invalid_rate']:.3f} steps={val['avg_steps']:.3f}"
            )
            score = self.validation_score(val)
            if score > best_val:
                best_val = score
                print(f"  [best] save best checkpoint ({self.best_metric}={best_val:.3f})")
                self.save_best()
            for adapter_name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
                self.model.save_lora(adapter_name, str(self.save_dir / f"{adapter_name}_step_{it + 1}"))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Plancraft Main/Sub agents with advantage GRPO.")
    parser.add_argument("--base-model", default="./models/qwen/Qwen2___5-1___5B-Instruct")
    parser.add_argument("--main-lora", default="./plancraft_mas_sft_50x1/main_agent")
    parser.add_argument("--sub-lora", default="./plancraft_mas_sft_50x1/sub_agent")
    parser.add_argument("--save-dir", default="./plancraft_mas_grpo_adv_smoke")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val.small.easy")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--val-tasks", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-response-len", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument(
        "--best-metric",
        choices=["success_rate", "best_success_rate", "reward", "best_reward", "valid_rate"],
        default="best_success_rate",
    )
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--advantage-clip", type=float, default=2.0)
    parser.add_argument("--min-advantage", type=float, default=0.01)
    parser.add_argument("--valid-weight", type=float, default=0.2)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_examples = load_examples(args.train_split, limit=args.tasks)
    val_examples = load_examples(args.val_split, limit=args.val_tasks)
    config = CoTrainConfig(
        base_model=args.base_model,
        main_lora_path=args.main_lora,
        sub_lora_path=args.sub_lora,
        save_dir=args.save_dir,
        group_size=args.group_size,
        max_response_len=args.max_response_len,
        lr=args.lr,
    )
    PlancraftMASGRPOTrainer(
        config,
        max_steps=args.max_steps,
        best_metric=args.best_metric,
        advantage_clip=args.advantage_clip,
        min_advantage=args.min_advantage,
        valid_weight=args.valid_weight,
        step_penalty=args.step_penalty,
        eval_samples=args.eval_samples,
    ).train(train_examples, val_examples, iterations=args.iterations)


if __name__ == "__main__":
    main()
