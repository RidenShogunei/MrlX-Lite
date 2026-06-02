"""GRPO-style training for the local mini deep-research environment."""

import argparse
import re
from pathlib import Path
from typing import List

from generate_research_sft_data import ANSWER_SYSTEM, MAIN_SYSTEM, SUB_SYSTEM
from grpo_v4 import CoTrainConfig, SharedModel
from research_environment import MiniResearchEnvironment, ResearchTask


class ResearchGRPOTrainer:
    def __init__(self, config: CoTrainConfig, research_steps: int = 3):
        self.config = config
        self.research_steps = research_steps
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    @staticmethod
    def history_text(history):
        if not history:
            return "No observations yet."
        lines = []
        for idx, (tool_call, observation) in enumerate(history, 1):
            lines.append(f"Step {idx} tool call: {tool_call}")
            lines.append(f"Step {idx} observation: {observation}")
        return "\n".join(lines)

    def build_main_prompt(self, task: ResearchTask, history) -> str:
        return self.model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": MAIN_SYSTEM},
                {"role": "user", "content": f"Question: {task.question}\nResearch history:\n{self.history_text(history)}"},
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

    def build_answer_prompt(self, task: ResearchTask, history) -> str:
        return self.model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": f"Question: {task.question}\nResearch history:\n{self.history_text(history)}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _extract_tool_call_block(text: str) -> str:
        match = re.search(r"\[tool_call\].*?\[/tool_call\]", text, re.DOTALL)
        return match.group(0) if match else text

    @staticmethod
    def _truncate_result(text: str) -> str:
        end = text.find("</result>")
        if end >= 0:
            return text[:end + len("</result>")]
        return text

    @staticmethod
    def _sub_reward(expected_output: str, response: str) -> float:
        expected = MiniResearchEnvironment.normalize(expected_output)
        got = MiniResearchEnvironment.normalize(response)
        return 1.0 if expected[:120] and expected[:120] in got else 0.0

    def generate_main_candidate(self, task: ResearchTask):
        history = []
        action_steps = []
        ok_any = False
        last_tool_output = ""
        for _step in range(self.research_steps):
            action_prompt = self.build_main_prompt(task, history)
            action_raw = self.model.generate_one(
                SharedModel.MAIN_ADAPTER,
                action_prompt,
                max_tokens=self.config.max_response_len,
                response_prefix="<thinking>",
                canonicalizer=None,
            )
            action_raw = self._truncate_result(action_raw)
            tool_call = self._extract_tool_call_block(action_raw)
            ok, tool_output = MiniResearchEnvironment.execute_tool(task, tool_call)
            if not ok:
                tool_output = "Tool execution failed"
            ok_any = ok_any or ok
            last_tool_output = tool_output
            history.append((tool_call, tool_output))
            action_steps.append((action_prompt, tool_call, ok, tool_output))

        answer_prompt = self.build_answer_prompt(task, history)
        answer_raw = self.model.generate_one(
            SharedModel.MAIN_ADAPTER,
            answer_prompt,
            max_tokens=self.config.max_response_len,
            response_prefix="<thinking>",
            canonicalizer=None,
        )
        answer_raw = self._truncate_result(answer_raw)
        combined = "".join(step[1] for step in action_steps) + answer_raw
        reward = MiniResearchEnvironment.reward(task, combined)
        return {
            "action_steps": action_steps,
            "answer_prompt": answer_prompt,
            "answer_raw": answer_raw,
            "raw": combined,
            "reward": reward["total"],
            "answer": reward["answer"],
            "evidence": reward["evidence"],
            "tool_valid": reward["tool_valid"],
            "tool_ok": 1.0 if ok_any else 0.0,
            "tool_output": last_tool_output,
        }

    def run_episode(self, task: ResearchTask):
        candidates = [self.generate_main_candidate(task) for _ in range(self.config.group_size)]
        candidates.sort(key=lambda c: (c["reward"], c["answer"], c["evidence"], c["tool_valid"]), reverse=True)
        best = candidates[0]

        sub_candidates = []
        if best["tool_ok"] > 0:
            for _, tool_call, ok, tool_output in best["action_steps"]:
                if not ok:
                    continue
                prompt_s = self.build_sub_prompt(tool_call)
                target = f"<thinking>execute tool</thinking><result>{tool_output}</result>"
                sub_candidates.append((prompt_s, target, 1.0))
            sub_candidates.sort(key=lambda x: x[2], reverse=True)

        return best, sub_candidates

    def evaluate(self, tasks: List[ResearchTask], samples: int = 2):
        if not tasks:
            return {"reward": 0.0, "answer": 0.0, "evidence": 0.0, "tool_valid": 0.0, "best_reward": 0.0}

        self.model.model.eval()
        rewards, answers, evidences, valids = [], [], [], []
        best_total = 0.0
        for task in tasks:
            task_best = 0.0
            for _ in range(samples):
                candidate = self.generate_main_candidate(task)
                rewards.append(candidate["reward"])
                answers.append(candidate["answer"])
                evidences.append(candidate["evidence"])
                valids.append(candidate["tool_valid"])
                task_best = max(task_best, candidate["reward"])
            best_total += task_best
        self.model.model.train()

        total = max(len(rewards), 1)
        return {
            "reward": sum(rewards) / total,
            "answer": sum(answers) / total,
            "evidence": sum(evidences) / total,
            "tool_valid": sum(valids) / total,
            "best_reward": best_total / len(tasks),
        }

    def train(self, train_tasks: List[ResearchTask], val_tasks: List[ResearchTask], iterations: int, eval_samples: int):
        print(f"[research-grpo] train={len(train_tasks)} val={len(val_tasks)} iter={iterations}")
        print(f"[research-grpo] lr={self.config.lr} group={self.config.group_size} threshold={self.config.reward_threshold}")

        self.model = SharedModel(self.config.base_model, self.config)
        self.model.load_sft_weights()
        self.model.model.train()

        print("\n===== Research Initial Validation =====")
        init_val = self.evaluate(val_tasks, samples=eval_samples)
        best_val = init_val["best_reward"]
        print(
            f"  [val:init] reward={init_val['reward']:.3f} best={init_val['best_reward']:.3f} "
            f"answer={init_val['answer']:.3f} evidence={init_val['evidence']:.3f} tool_valid={init_val['tool_valid']:.3f}"
        )
        print(f"  [best] save initial checkpoint (best_reward={best_val:.3f})")
        for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
            self.model.save_lora(name, str(self.save_dir / "best" / name))

        for it in range(iterations):
            print(f"\n===== Research Iter {it + 1}/{iterations} =====")
            rewards, answers, evidences, valids, sub_rewards = [], [], [], [], []
            updates, sub_updates = 0, 0

            for task in train_tasks:
                best, sub_candidates = self.run_episode(task)
                rewards.append(best["reward"])
                answers.append(best["answer"])
                evidences.append(best["evidence"])
                valids.append(best["tool_valid"])

                if best["reward"] >= self.config.reward_threshold:
                    for prompt, action, _ok, _tool_output in best["action_steps"]:
                        self.model.sft_step(
                            SharedModel.MAIN_ADAPTER, prompt, action, weight=best["reward"]
                        )
                    self.model.sft_step(
                        SharedModel.MAIN_ADAPTER,
                        best["answer_prompt"],
                        best["answer_raw"],
                        weight=best["reward"],
                    )
                    updates += 1

                if sub_candidates:
                    prompt_s, resp_s, rew_s = sub_candidates[0]
                    sub_rewards.append(rew_s)
                    if rew_s >= self.config.reward_threshold:
                        self.model.sft_step(SharedModel.SUB_ADAPTER, prompt_s, resp_s, weight=rew_s)
                        sub_updates += 1

            print(
                f"  train reward={sum(rewards)/max(len(rewards),1):.3f} "
                f"answer={sum(answers)/max(len(answers),1):.3f} "
                f"evidence={sum(evidences)/max(len(evidences),1):.3f} "
                f"tool_valid={sum(valids)/max(len(valids),1):.3f} updates={updates}/{len(train_tasks)} | "
                f"sub={sum(sub_rewards)/max(len(sub_rewards),1):.3f} sub_updates={sub_updates}"
            )

            val = self.evaluate(val_tasks, samples=eval_samples)
            print(
                f"  [val] reward={val['reward']:.3f} best={val['best_reward']:.3f} "
                f"answer={val['answer']:.3f} evidence={val['evidence']:.3f} tool_valid={val['tool_valid']:.3f}"
            )
            if val["best_reward"] > best_val:
                best_val = val["best_reward"]
                print(f"  [best] save best checkpoint (best_reward={best_val:.3f})")
                for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
                    self.model.save_lora(name, str(self.save_dir / "best" / name))

            for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
                self.model.save_lora(name, str(self.save_dir / f"{name}_step_{it + 1}"))

        print("\n[OK] research GRPO complete")


def parse_args():
    parser = argparse.ArgumentParser(description="Train mini research Main/Sub agents with GRPO-style updates.")
    parser.add_argument("--base-model", default="./models/qwen/Qwen2___5-1___5B-Instruct")
    parser.add_argument("--sft-dir", default="./research_sft_checkpoints")
    parser.add_argument("--save-dir", default="./research_grpo_smoke")
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--val-tasks", type=int, default=20)
    parser.add_argument("--test-tasks", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--max-response-len", type=int, default=192)
    parser.add_argument("--research-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--reward-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    train_tasks, val_tasks, _ = MiniResearchEnvironment.split(args.tasks, args.val_tasks, args.test_tasks, seed=args.seed)
    config = CoTrainConfig(
        base_model=args.base_model,
        sft_dir=args.sft_dir,
        save_dir=args.save_dir,
        group_size=args.group_size,
        max_response_len=args.max_response_len,
        lr=args.lr,
        reward_threshold=args.reward_threshold,
    )
    ResearchGRPOTrainer(config, research_steps=args.research_steps).train(
        train_tasks, val_tasks, iterations=args.iterations, eval_samples=args.eval_samples
    )


if __name__ == "__main__":
    main()
