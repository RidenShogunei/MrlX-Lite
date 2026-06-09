"""Advantage-style GRPO for Plancraft Main/Sub agents."""

import argparse
import json
import random
import re
from pathlib import Path

import torch

from analyze_hotpotqa_mas_results import build_prompt
from grpo_v4 import CoTrainConfig, SharedModel
from plancraft_environment import PlancraftBenchEpisode, flatten_subplans, load_examples
from plancraft_prompts import MAIN_SYSTEM, STRUCTURED_SUB_SYSTEM, SUB_SYSTEM, history_text


def first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def normalize_action(text: str) -> str:
    return re.sub(r"\s+", " ", first_line(text).lower()).strip()


def extract_structured_action(text: str) -> str:
    match = re.search(r"<action>(.*?)</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return first_line(match.group(1))
    return first_line(text)


def truncate_structured_sub(text: str) -> str:
    text = text.strip()
    end = text.find("</action>")
    if end >= 0:
        return text[: end + len("</action>")].strip()
    return "\n".join(text.splitlines()[:3]).strip()


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
        main_success_weight: float = 1.0,
        main_valid_weight: float = 0.2,
        main_oracle_weight: float = 0.2,
        main_progress_weight: float = 0.4,
        sub_global_weight: float = 0.5,
        sub_valid_weight: float = 0.3,
        sub_oracle_weight: float = 0.5,
        sub_progress_weight: float = 0.6,
        sub_agreement_weight: float = 0.2,
        repeat_action_penalty: float = 0.2,
        incorrect_stop_penalty: float = 0.5,
        structured_sub: bool = False,
        reward_gap_threshold: float = 0.02,
        sft_replay_path: str | None = None,
        sft_replay_per_group: int = 0,
        sft_replay_weight: float = 0.1,
        rollout_temperature: float = 0.8,
        eval_temperature: float = 0.2,
        eval_top_p: float = 0.9,
        eval_repetition_penalty: float = 1.05,
        eval_max_steps: int = 10,
        eval_seed: int = 123,
        train_main: bool = True,
        train_sub: bool = True,
        policy_clip: float = 0.2,
        kl_beta: float = 0.01,
        policy_epochs: int = 2,
    ):
        self.config = config
        self.max_steps = max_steps
        self.best_metric = best_metric
        self.advantage_clip = advantage_clip
        self.min_advantage = min_advantage
        self.valid_weight = valid_weight
        self.step_penalty = step_penalty
        self.eval_samples = max(eval_samples, 1)
        self.main_success_weight = main_success_weight
        self.main_valid_weight = main_valid_weight
        self.main_oracle_weight = main_oracle_weight
        self.main_progress_weight = main_progress_weight
        self.sub_global_weight = sub_global_weight
        self.sub_valid_weight = sub_valid_weight
        self.sub_oracle_weight = sub_oracle_weight
        self.sub_progress_weight = sub_progress_weight
        self.sub_agreement_weight = sub_agreement_weight
        self.repeat_action_penalty = repeat_action_penalty
        self.incorrect_stop_penalty = incorrect_stop_penalty
        self.structured_sub = structured_sub
        self.reward_gap_threshold = max(reward_gap_threshold, 0.0)
        self.sft_replay_path = sft_replay_path
        self.sft_replay_per_group = max(sft_replay_per_group, 0)
        self.sft_replay_weight = max(sft_replay_weight, 0.0)
        self.rollout_temperature = rollout_temperature
        self.eval_temperature = eval_temperature
        self.eval_top_p = eval_top_p
        self.eval_repetition_penalty = eval_repetition_penalty
        self.eval_max_steps = eval_max_steps
        self.eval_seed = eval_seed
        self.train_main = train_main
        self.train_sub = train_sub
        self.policy_clip = policy_clip
        self.kl_beta = kl_beta
        self.policy_epochs = max(policy_epochs, 1)
        self.replay_samples = {SharedModel.MAIN_ADAPTER: [], SharedModel.SUB_ADAPTER: []}
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def generate_action(self, adapter_name: str, prompt: str, structured: bool = False, evaluation: bool = False) -> str:
        text = self.model.generate_one(
            adapter_name,
            prompt,
            max_tokens=self.config.max_response_len,
            response_prefix="",
            canonicalizer=truncate_structured_sub if structured else first_line,
            temperature=self.eval_temperature if evaluation else self.rollout_temperature,
            top_p=self.eval_top_p if evaluation else 0.95,
            repetition_penalty=self.eval_repetition_penalty if evaluation else 1.0,
        )
        return text

    def build_sub_prompt(self, observation: str, history) -> str:
        return build_prompt(
            self.model.tokenizer,
            STRUCTURED_SUB_SYSTEM if self.structured_sub else SUB_SYSTEM,
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

    @staticmethod
    def action_is_valid(episode: PlancraftBenchEpisode, action: str) -> bool:
        try:
            return not isinstance(episode.wrapper.parse_raw_model_response(action), str)
        except Exception:
            return False

    @staticmethod
    def oracle_next_action(episode: PlancraftBenchEpisode) -> str:
        try:
            actions = flatten_subplans(episode.oracle_subplans())
        except Exception:
            return ""
        return first_line(actions[0]) if actions else ""

    @staticmethod
    def oracle_remaining_steps(episode: PlancraftBenchEpisode) -> int:
        if episode.terminated or episode.truncated or bool(episode.wrapper.success):
            return 0
        try:
            return len(flatten_subplans(episode.oracle_subplans()))
        except Exception:
            return 0

    @staticmethod
    def action_matches_oracle(action: str, oracle_action: str) -> bool:
        if not action or not oracle_action:
            return False
        return normalize_action(action) == normalize_action(oracle_action)

    def candidate_rewards(self, result, step_scores) -> tuple[float, float]:
        return (
            sum(score["main_step_reward"] for score in step_scores),
            sum(score["sub_step_reward"] for score in step_scores),
        )

    def step_rewards(
        self,
        *,
        main_valid: float,
        sub_valid: float,
        main_oracle: float,
        sub_oracle: float,
        progress: float,
        agreement: float,
        repeated_action: float,
        terminal_success: float,
        incorrect_stop: float,
    ) -> tuple[float, float]:
        main_reward = (
            self.main_success_weight * terminal_success
            + self.main_valid_weight * main_valid
            + self.main_oracle_weight * main_oracle
            + self.main_progress_weight * progress
            - self.repeat_action_penalty * repeated_action
            - self.incorrect_stop_penalty * incorrect_stop
            - self.step_penalty
        )
        sub_reward = (
            self.sub_global_weight * terminal_success
            + self.sub_valid_weight * sub_valid
            + self.sub_oracle_weight * sub_oracle
            + self.sub_progress_weight * progress
            + self.sub_agreement_weight * agreement
            - self.repeat_action_penalty * repeated_action
            - self.incorrect_stop_penalty * incorrect_stop
            - self.step_penalty
        )
        return main_reward, sub_reward

    def generate_candidate(self, example, evaluation: bool = False):
        max_steps = self.eval_max_steps if evaluation else self.max_steps
        episode = PlancraftBenchEpisode(example, max_steps=max_steps)
        observation = episode.reset()
        history = []
        previous_main_actions = set()
        steps = []
        step_scores = []
        for _step in range(max_steps):
            oracle_action = self.oracle_next_action(episode)
            oracle_steps_before = self.oracle_remaining_steps(episode)
            sub_prompt = self.build_sub_prompt(observation, history)
            sub_raw = self.generate_action(
                SharedModel.SUB_ADAPTER,
                sub_prompt,
                structured=self.structured_sub,
                evaluation=evaluation,
            )
            sub_old_logprobs = None
            sub_reference_logprobs = None
            if not evaluation:
                sub_old_logprobs = self.model.response_token_logprobs(
                    SharedModel.SUB_ADAPTER,
                    sub_prompt,
                    sub_raw,
                )
                sub_reference_logprobs = self.model.response_token_logprobs(
                    self.model.reference_adapter(SharedModel.SUB_ADAPTER),
                    sub_prompt,
                    sub_raw,
                )
            main_prompt = self.build_main_prompt(observation, history, sub_raw)
            main_raw = self.generate_action(SharedModel.MAIN_ADAPTER, main_prompt, evaluation=evaluation)
            main_old_logprobs = None
            main_reference_logprobs = None
            if not evaluation:
                main_old_logprobs = self.model.response_token_logprobs(
                    SharedModel.MAIN_ADAPTER,
                    main_prompt,
                    main_raw,
                )
                main_reference_logprobs = self.model.response_token_logprobs(
                    self.model.reference_adapter(SharedModel.MAIN_ADAPTER),
                    main_prompt,
                    main_raw,
                )
            sub_action = extract_structured_action(sub_raw) if self.structured_sub else sub_raw
            sub_norm = normalize_action(sub_action)
            main_norm = normalize_action(main_raw)
            oracle_norm = normalize_action(oracle_action)
            sub_valid = 1.0 if self.action_is_valid(episode, sub_action) else 0.0
            main_valid = 1.0 if self.action_is_valid(episode, main_raw) else 0.0
            observation, _reward, terminated, truncated, _info = episode.step(main_raw)
            oracle_steps_after = self.oracle_remaining_steps(episode)
            progress = 0.0
            if main_valid and oracle_steps_before > 0:
                progress = max(min((oracle_steps_before - oracle_steps_after) / oracle_steps_before, 1.0), -1.0)
            sub_oracle_match = 1.0 if sub_norm and sub_norm == oracle_norm else 0.0
            main_oracle_match = 1.0 if main_norm and main_norm == oracle_norm else 0.0
            agreement = 1.0 if sub_norm and sub_norm == main_norm else 0.0
            repeated_action = 1.0 if main_norm and main_norm in previous_main_actions else 0.0
            terminal_success = 1.0 if terminated and bool(episode.wrapper.success) else 0.0
            incorrect_stop = 1.0 if terminated and not terminal_success and main_norm.startswith("impossible:") else 0.0
            main_step_reward, sub_step_reward = self.step_rewards(
                main_valid=main_valid,
                sub_valid=sub_valid,
                main_oracle=main_oracle_match,
                sub_oracle=sub_oracle_match,
                progress=progress,
                agreement=agreement,
                repeated_action=repeated_action,
                terminal_success=terminal_success,
                incorrect_stop=incorrect_stop,
            )
            score = {
                "sub_valid": sub_valid,
                "main_valid": main_valid,
                "sub_oracle_match": sub_oracle_match,
                "main_oracle_match": main_oracle_match,
                "sub_main_agreement": agreement,
                "oracle_progress": progress,
                "repeated_action": repeated_action,
                "terminal_success": terminal_success,
                "incorrect_stop": incorrect_stop,
                "main_step_reward": main_step_reward,
                "sub_step_reward": sub_step_reward,
            }
            step_scores.append(score)
            steps.append(
                {
                    "sub_prompt": sub_prompt,
                    "sub_raw": sub_raw,
                    "main_prompt": main_prompt,
                    "main_raw": main_raw,
                    "oracle_action": oracle_action,
                    "sub_old_logprobs": sub_old_logprobs,
                    "sub_reference_logprobs": sub_reference_logprobs,
                    "main_old_logprobs": main_old_logprobs,
                    "main_reference_logprobs": main_reference_logprobs,
                    **score,
                }
            )
            if main_norm:
                previous_main_actions.add(main_norm)
            history.append((sub_raw, main_raw, observation))
            if terminated or truncated:
                break
        result = episode.result()
        valid_rate = result.valid_action_count / max(result.action_count, 1)
        main_reward, sub_reward = self.candidate_rewards(result, step_scores)
        return {
            "steps": steps,
            "result": result,
            "reward": main_reward,
            "main_reward": main_reward,
            "sub_reward": sub_reward,
            "success": 1.0 if result.success else 0.0,
            "valid_rate": valid_rate,
            "invalid_rate": result.invalid_action_count / max(result.action_count, 1),
            "env_reward": result.reward,
            "step_count": result.steps,
            "main_valid": self.average(step_scores, "main_valid"),
            "sub_valid": self.average(step_scores, "sub_valid"),
            "main_oracle_match": self.average(step_scores, "main_oracle_match"),
            "sub_oracle_match": self.average(step_scores, "sub_oracle_match"),
            "sub_main_agreement": self.average(step_scores, "sub_main_agreement"),
            "oracle_progress": self.average(step_scores, "oracle_progress"),
            "repeat_rate": self.average(step_scores, "repeated_action"),
            "incorrect_stop_rate": self.average(step_scores, "incorrect_stop"),
        }

    def group_advantages(self, candidates, reward_key: str, advantage_key: str):
        values = [cand[reward_key] for cand in candidates]
        reward_gap = max(values) - min(values) if values else 0.0
        if reward_gap < self.reward_gap_threshold:
            for cand in candidates:
                cand[advantage_key] = 0.0
            return candidates
        mean = sum(values) / max(len(values), 1)
        var = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        std = max(var ** 0.5, 1e-6)
        for cand, value in zip(candidates, values):
            adv = (value - mean) / std
            cand[advantage_key] = max(min(adv, self.advantage_clip), -self.advantage_clip)
        return candidates

    def run_group(self, example):
        candidates = [self.generate_candidate(example) for _ in range(self.config.group_size)]
        self.group_advantages(candidates, "main_reward", "main_advantage")
        self.group_advantages(candidates, "sub_reward", "sub_advantage")
        self.assign_step_advantages(candidates, "main_step_reward", "main_step_advantage")
        self.assign_step_advantages(candidates, "sub_step_reward", "sub_step_advantage")
        first_actions = [normalize_action(candidate["steps"][0]["main_raw"]) for candidate in candidates if candidate["steps"]]
        unique_actions = len(set(first_actions))
        main_step_advantages = [
            abs(step.get("main_step_advantage", 0.0)) for candidate in candidates for step in candidate["steps"]
        ]
        sub_step_advantages = [
            abs(step.get("sub_step_advantage", 0.0)) for candidate in candidates for step in candidate["steps"]
        ]
        for candidate in candidates:
            candidate["group_unique_first_action_rate"] = unique_actions / max(len(first_actions), 1)
            candidate["main_zero_advantage_rate"] = sum(
                advantage < self.min_advantage for advantage in main_step_advantages
            ) / max(len(main_step_advantages), 1)
            candidate["sub_zero_advantage_rate"] = sum(
                advantage < self.min_advantage for advantage in sub_step_advantages
            ) / max(len(sub_step_advantages), 1)
        candidates.sort(key=lambda cand: (cand["main_reward"], cand["success"], cand["valid_rate"]), reverse=True)
        return candidates

    def assign_step_advantages(self, candidates, reward_key: str, advantage_key: str):
        max_steps = max((len(candidate["steps"]) for candidate in candidates), default=0)
        for step_index in range(max_steps):
            available = [candidate["steps"][step_index] for candidate in candidates if step_index < len(candidate["steps"])]
            values = [step[reward_key] for step in available]
            reward_gap = max(values) - min(values) if values else 0.0
            if len(values) < 2 or reward_gap < self.reward_gap_threshold:
                for step in available:
                    step[advantage_key] = 0.0
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = max(variance**0.5, 1e-6)
            for step, value in zip(available, values):
                advantage = (value - mean) / std
                step[advantage_key] = max(min(advantage, self.advantage_clip), -self.advantage_clip)

    def load_replay_samples(self):
        if not self.sft_replay_path or self.sft_replay_per_group <= 0:
            return
        with open(self.sft_replay_path, "r", encoding="utf-8") as replay_file:
            for line in replay_file:
                item = json.loads(line)
                messages = item.get("messages", [])
                if not messages or messages[-1].get("role") != "assistant":
                    continue
                category = item.get("category")
                adapter_name = SharedModel.MAIN_ADAPTER if category == "main" else SharedModel.SUB_ADAPTER
                prompt = self.model.tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                self.replay_samples[adapter_name].append((prompt, messages[-1]["content"]))
        print(
            "[plancraft-mas-grpo] replay samples "
            f"main={len(self.replay_samples[SharedModel.MAIN_ADAPTER])} "
            f"sub={len(self.replay_samples[SharedModel.SUB_ADAPTER])}"
        )

    def replay_batch(self, adapter_name: str):
        samples = self.replay_samples[adapter_name]
        if not samples or self.sft_replay_per_group <= 0:
            return []
        count = min(self.sft_replay_per_group, len(samples))
        return random.sample(samples, count)

    def apply_adapter_update(self, adapter_name: str, records, replay_records) -> tuple[int, dict]:
        if not records and not replay_records:
            return 0, {
                "policy_loss": 0.0,
                "kl": 0.0,
                "ratio": 1.0,
                "clip_fraction": 0.0,
                "tokens": 0,
            }
        policy_loss_total = 0.0
        kl_total = 0.0
        ratio_total = 0.0
        clip_fraction_total = 0.0
        policy_records = 0
        token_total = 0
        optimizer_steps = 0
        for _epoch in range(self.policy_epochs):
            self.model.optimizer_zero_grad(adapter_name)
            backward_count = 0
            rollout_scale = 1.0 / max(len(records), 1)
            for prompt, response, old_logprobs, reference_logprobs, advantage in records:
                stats = self.model.grpo_backward(
                    adapter_name,
                    prompt,
                    response,
                    old_logprobs,
                    reference_logprobs,
                    advantage=advantage,
                    policy_clip=self.policy_clip,
                    kl_beta=self.kl_beta,
                    weight=rollout_scale,
                )
                if stats["tokens"] > 0:
                    backward_count += 1
                    policy_records += 1
                    policy_loss_total += stats["policy_loss"]
                    kl_total += stats["kl"]
                    ratio_total += stats["ratio"]
                    clip_fraction_total += stats["clip_fraction"]
                    token_total += stats["tokens"]
            replay_scale = self.sft_replay_weight / max(len(replay_records), 1)
            for prompt, response in replay_records:
                loss = self.model.sft_backward(
                    adapter_name,
                    prompt,
                    response,
                    weight=replay_scale,
                )
                backward_count += 1 if loss != 0.0 else 0
            if backward_count == 0:
                self.model.optimizer_zero_grad(adapter_name)
                continue
            self.model.optimizer_step(adapter_name)
            optimizer_steps += 1
        return optimizer_steps, {
            "policy_loss": policy_loss_total / max(policy_records, 1),
            "kl": kl_total / max(policy_records, 1),
            "ratio": ratio_total / max(policy_records, 1),
            "clip_fraction": clip_fraction_total / max(policy_records, 1),
            "tokens": token_total,
        }

    def apply_advantage_update(self, candidates):
        main_records = []
        sub_records = []
        for cand in candidates:
            for step in cand["steps"]:
                main_adv = step.get("main_step_advantage", 0.0)
                sub_adv = step.get("sub_step_advantage", 0.0)
                if abs(sub_adv) >= self.min_advantage:
                    sub_records.append(
                        (
                            step["sub_prompt"],
                            step["sub_raw"],
                            step["sub_old_logprobs"],
                            step["sub_reference_logprobs"],
                            sub_adv,
                        )
                    )
                if abs(main_adv) >= self.min_advantage:
                    main_records.append(
                        (
                            step["main_prompt"],
                            step["main_raw"],
                            step["main_old_logprobs"],
                            step["main_reference_logprobs"],
                            main_adv,
                        )
                    )
        sub_updates = 0
        sub_stats = {"policy_loss": 0.0, "kl": 0.0, "ratio": 1.0, "clip_fraction": 0.0, "tokens": 0}
        if self.train_sub:
            sub_updates, sub_stats = self.apply_adapter_update(
                SharedModel.SUB_ADAPTER,
                sub_records,
                self.replay_batch(SharedModel.SUB_ADAPTER),
            )
        main_updates = 0
        main_stats = {"policy_loss": 0.0, "kl": 0.0, "ratio": 1.0, "clip_fraction": 0.0, "tokens": 0}
        if self.train_main:
            main_updates, main_stats = self.apply_adapter_update(
                SharedModel.MAIN_ADAPTER,
                main_records,
                self.replay_batch(SharedModel.MAIN_ADAPTER),
            )
        return main_updates, sub_updates, main_stats, sub_stats

    @staticmethod
    def average(rows, key: str) -> float:
        return sum(row[key] for row in rows) / max(len(rows), 1)

    def evaluate(self, examples):
        self.model.model.eval()
        random.seed(self.eval_seed)
        torch.manual_seed(self.eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.eval_seed)
        rows = []
        for example in examples:
            samples = [self.generate_candidate(example, evaluation=True) for _ in range(self.eval_samples)]
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
            "main_reward": self.average(rows, "main_reward"),
            "sub_reward": self.average(rows, "sub_reward"),
            "env_reward": self.average(rows, "env_reward"),
            "valid_rate": self.average(rows, "valid_rate"),
            "invalid_rate": self.average(rows, "invalid_rate"),
            "main_valid": self.average(rows, "main_valid"),
            "sub_valid": self.average(rows, "sub_valid"),
            "main_oracle_match": self.average(rows, "main_oracle_match"),
            "sub_oracle_match": self.average(rows, "sub_oracle_match"),
            "sub_main_agreement": self.average(rows, "sub_main_agreement"),
            "oracle_progress": self.average(rows, "oracle_progress"),
            "repeat_rate": self.average(rows, "repeat_rate"),
            "incorrect_stop_rate": self.average(rows, "incorrect_stop_rate"),
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
        print(
            "[plancraft-mas-grpo] reward weights "
            f"main=(success:{self.main_success_weight}, valid:{self.main_valid_weight}, "
            f"oracle:{self.main_oracle_weight}, progress:{self.main_progress_weight}) "
            f"sub=(global:{self.sub_global_weight}, valid:{self.sub_valid_weight}, "
            f"oracle:{self.sub_oracle_weight}, progress:{self.sub_progress_weight}, "
            f"agree:{self.sub_agreement_weight})"
        )
        print(
            f"[plancraft-mas-grpo] step penalties repeat={self.repeat_action_penalty} "
            f"incorrect_stop={self.incorrect_stop_penalty}"
        )
        print(
            f"[plancraft-mas-grpo] eval_samples={self.eval_samples} "
            f"structured_sub={self.structured_sub} reward_gap_threshold={self.reward_gap_threshold}"
        )
        print(
            f"[plancraft-mas-grpo] rollout_temperature={self.rollout_temperature} "
            f"eval_temperature={self.eval_temperature} eval_top_p={self.eval_top_p} "
            f"eval_max_steps={self.eval_max_steps} eval_seed={self.eval_seed}"
        )
        print(
            f"[plancraft-mas-grpo] replay_path={self.sft_replay_path} "
            f"replay_per_group={self.sft_replay_per_group} replay_weight={self.sft_replay_weight}"
        )
        print(f"[plancraft-mas-grpo] train_main={self.train_main} train_sub={self.train_sub}")
        print(
            f"[plancraft-mas-grpo] policy_clip={self.policy_clip} "
            f"kl_beta={self.kl_beta} policy_epochs={self.policy_epochs}"
        )
        self.model = SharedModel(self.config.base_model, self.config)
        self.model.load_sft_weights()
        self.load_replay_samples()
        self.model.model.train()

        init = self.evaluate(val_examples)
        best_val = self.validation_score(init)
        print(
            f"  [val:init] success={init['success_rate']:.3f} reward={init['reward']:.3f} "
            f"best_success={init['best_success_rate']:.3f} best_reward={init['best_reward']:.3f} "
            f"main_reward={init['main_reward']:.3f} sub_reward={init['sub_reward']:.3f} "
            f"env_reward={init['env_reward']:.3f} valid={init['valid_rate']:.3f} "
            f"invalid={init['invalid_rate']:.3f} steps={init['avg_steps']:.3f} "
            f"main_oracle={init['main_oracle_match']:.3f} sub_oracle={init['sub_oracle_match']:.3f} "
            f"progress={init['oracle_progress']:.3f}"
        )
        self.save_best()

        for it in range(iterations):
            print(f"\n===== Plancraft MAS GRPO Iter {it + 1}/{iterations} =====")
            rows = []
            main_updates, sub_updates = 0, 0
            main_policy_losses, sub_policy_losses = [], []
            main_kls, sub_kls = [], []
            main_ratios, sub_ratios = [], []
            main_clip_fractions, sub_clip_fractions = [], []
            for example in train_examples:
                candidates = self.run_group(example)
                rows.append(candidates[0])
                u_main, u_sub, main_stats, sub_stats = self.apply_advantage_update(candidates)
                main_updates += u_main
                sub_updates += u_sub
                if main_stats["tokens"]:
                    main_policy_losses.append(main_stats["policy_loss"])
                    main_kls.append(main_stats["kl"])
                    main_ratios.append(main_stats["ratio"])
                    main_clip_fractions.append(main_stats["clip_fraction"])
                if sub_stats["tokens"]:
                    sub_policy_losses.append(sub_stats["policy_loss"])
                    sub_kls.append(sub_stats["kl"])
                    sub_ratios.append(sub_stats["ratio"])
                    sub_clip_fractions.append(sub_stats["clip_fraction"])
            print(
                f"  train success={self.average(rows, 'success'):.3f} "
                f"main_reward={self.average(rows, 'main_reward'):.3f} "
                f"sub_reward={self.average(rows, 'sub_reward'):.3f} "
                f"valid={self.average(rows, 'valid_rate'):.3f} "
                f"invalid={self.average(rows, 'invalid_rate'):.3f} "
                f"steps={self.average(rows, 'step_count'):.3f} "
                f"main_oracle={self.average(rows, 'main_oracle_match'):.3f} "
                f"sub_oracle={self.average(rows, 'sub_oracle_match'):.3f} "
                f"progress={self.average(rows, 'oracle_progress'):.3f} "
                f"agree={self.average(rows, 'sub_main_agreement'):.3f} "
                f"repeat={self.average(rows, 'repeat_rate'):.3f} "
                f"incorrect_stop={self.average(rows, 'incorrect_stop_rate'):.3f} "
                f"unique_first={self.average(rows, 'group_unique_first_action_rate'):.3f} "
                f"zero_adv main={self.average(rows, 'main_zero_advantage_rate'):.3f} "
                f"sub={self.average(rows, 'sub_zero_advantage_rate'):.3f} "
                f"updates main={main_updates} sub={sub_updates} "
                f"policy_loss main={sum(main_policy_losses) / max(len(main_policy_losses), 1):.4f} "
                f"sub={sum(sub_policy_losses) / max(len(sub_policy_losses), 1):.4f} "
                f"kl main={sum(main_kls) / max(len(main_kls), 1):.6f} "
                f"sub={sum(sub_kls) / max(len(sub_kls), 1):.6f} "
                f"ratio main={sum(main_ratios) / max(len(main_ratios), 1):.4f} "
                f"sub={sum(sub_ratios) / max(len(sub_ratios), 1):.4f} "
                f"clip_frac main={sum(main_clip_fractions) / max(len(main_clip_fractions), 1):.4f} "
                f"sub={sum(sub_clip_fractions) / max(len(sub_clip_fractions), 1):.4f}"
            )
            val = self.evaluate(val_examples)
            print(
                f"  [val] success={val['success_rate']:.3f} reward={val['reward']:.3f} "
                f"best_success={val['best_success_rate']:.3f} best_reward={val['best_reward']:.3f} "
                f"main_reward={val['main_reward']:.3f} sub_reward={val['sub_reward']:.3f} "
                f"env_reward={val['env_reward']:.3f} valid={val['valid_rate']:.3f} "
                f"invalid={val['invalid_rate']:.3f} steps={val['avg_steps']:.3f} "
                f"main_oracle={val['main_oracle_match']:.3f} sub_oracle={val['sub_oracle_match']:.3f} "
                f"progress={val['oracle_progress']:.3f}"
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
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--val-tasks", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-response-len", type=int, default=50)
    parser.add_argument("--max-train-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument(
        "--best-metric",
        choices=["success_rate", "best_success_rate", "reward", "best_reward", "valid_rate"],
        default="success_rate",
    )
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--advantage-clip", type=float, default=2.0)
    parser.add_argument("--min-advantage", type=float, default=0.01)
    parser.add_argument("--valid-weight", type=float, default=0.2)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--main-success-weight", type=float, default=1.0)
    parser.add_argument("--main-valid-weight", type=float, default=0.2)
    parser.add_argument("--main-oracle-weight", type=float, default=0.2)
    parser.add_argument("--main-progress-weight", type=float, default=0.4)
    parser.add_argument("--sub-global-weight", type=float, default=0.5)
    parser.add_argument("--sub-valid-weight", type=float, default=0.3)
    parser.add_argument("--sub-oracle-weight", type=float, default=0.5)
    parser.add_argument("--sub-progress-weight", type=float, default=0.6)
    parser.add_argument("--sub-agreement-weight", type=float, default=0.0)
    parser.add_argument("--repeat-action-penalty", type=float, default=0.2)
    parser.add_argument("--incorrect-stop-penalty", type=float, default=0.5)
    parser.add_argument("--structured-sub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reward-gap-threshold", type=float, default=0.02)
    parser.add_argument("--sft-replay-path", default=None)
    parser.add_argument("--sft-replay-per-group", type=int, default=0)
    parser.add_argument("--sft-replay-weight", type=float, default=0.1)
    parser.add_argument("--rollout-temperature", type=float, default=0.8)
    parser.add_argument("--eval-temperature", type=float, default=0.2)
    parser.add_argument("--eval-top-p", type=float, default=0.9)
    parser.add_argument("--eval-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--eval-max-steps", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--train-main", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-sub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-clip", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--policy-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_examples = load_examples(args.train_split, offset=args.train_offset, limit=args.tasks)
    val_examples = load_examples(args.val_split, offset=args.val_offset, limit=args.val_tasks)
    config = CoTrainConfig(
        base_model=args.base_model,
        main_lora_path=args.main_lora,
        sub_lora_path=args.sub_lora,
        save_dir=args.save_dir,
        group_size=args.group_size,
        max_response_len=args.max_response_len,
        max_train_length=args.max_train_length,
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
        main_success_weight=args.main_success_weight,
        main_valid_weight=args.main_valid_weight,
        main_oracle_weight=args.main_oracle_weight,
        main_progress_weight=args.main_progress_weight,
        sub_global_weight=args.sub_global_weight,
        sub_valid_weight=args.sub_valid_weight,
        sub_oracle_weight=args.sub_oracle_weight,
        sub_progress_weight=args.sub_progress_weight,
        sub_agreement_weight=args.sub_agreement_weight,
        repeat_action_penalty=args.repeat_action_penalty,
        incorrect_stop_penalty=args.incorrect_stop_penalty,
        structured_sub=args.structured_sub,
        reward_gap_threshold=args.reward_gap_threshold,
        sft_replay_path=args.sft_replay_path,
        sft_replay_per_group=args.sft_replay_per_group,
        sft_replay_weight=args.sft_replay_weight,
        rollout_temperature=args.rollout_temperature,
        eval_temperature=args.eval_temperature,
        eval_top_p=args.eval_top_p,
        eval_repetition_penalty=args.eval_repetition_penalty,
        eval_max_steps=args.eval_max_steps,
        eval_seed=args.eval_seed,
        train_main=args.train_main,
        train_sub=args.train_sub,
        policy_clip=args.policy_clip,
        kl_beta=args.kl_beta,
        policy_epochs=args.policy_epochs,
    ).train(train_examples, val_examples, iterations=args.iterations)


if __name__ == "__main__":
    main()
