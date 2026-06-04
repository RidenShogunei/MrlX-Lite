"""Plancraft benchmark wrapper for Main/Sub planning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def import_plancraft():
    try:
        from plancraft.environment.planner import get_subplans
        from plancraft.simple import PlancraftGymWrapper, get_plancraft_examples

        return PlancraftGymWrapper, get_plancraft_examples, get_subplans
    except KeyError as exc:
        if str(exc).strip("'\"") == "acacia_logs":
            raise RuntimeError(
                "Plancraft failed to import on Windows due to its tag path parser. "
                "Run `python patch_plancraft_windows.py` once, then retry."
            ) from exc
        raise


@dataclass
class PlancraftRunResult:
    task_id: str
    target: str
    impossible: bool
    success: bool
    reward: float
    steps: int
    optimal_path_length: int | None
    action_count: int
    valid_action_count: int
    invalid_action_count: int
    terminated: bool
    truncated: bool
    reason: str

    @property
    def efficiency(self) -> float:
        if not self.success or not self.optimal_path_length or self.steps <= 0:
            return 0.0
        return min(self.optimal_path_length / self.steps, 1.0)


def load_examples(split: str, offset: int = 0, limit: int | None = None):
    _wrapper, get_examples, _subplans = import_plancraft()
    examples = get_examples(split)
    end = None if limit is None else offset + limit
    return examples[offset:end]


class PlancraftBenchEpisode:
    """Small adapter around PlancraftGymWrapper.

    The wrapper keeps Plancraft's official transition/reward logic intact while
    exposing oracle subplans for teacher-forcing and sanity checks.
    """

    def __init__(self, example, max_steps: int = 30):
        PlancraftGymWrapper, _get_examples, get_subplans = import_plancraft()
        self.get_subplans = get_subplans
        self.wrapper = PlancraftGymWrapper(example=example, max_steps=max_steps, use_text_inventory=True)
        self.example = example
        self.observation: dict[str, Any] | None = None
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        self.info: dict[str, Any] = {}
        self.action_count = 0
        self.valid_action_count = 0
        self.invalid_action_count = 0

    def reset(self) -> str:
        self.observation, self.reward, self.terminated, self.truncated, self.info = self.wrapper.step()
        return self.observation["text"]

    def oracle_subplans(self) -> list[list[str]]:
        if self.observation is None:
            self.reset()
        subplans, _plan = self.get_subplans(self.observation)
        return subplans

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        parsed_action = self.wrapper.parse_raw_model_response(action)
        action_is_valid = not isinstance(parsed_action, str)
        obs, reward, terminated, truncated, info = self.wrapper.step(action)
        self.observation = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.action_count += 1
        if action_is_valid:
            self.valid_action_count += 1
        else:
            self.invalid_action_count += 1
        return obs.get("text", ""), reward, terminated, truncated, info

    def result(self) -> PlancraftRunResult:
        return PlancraftRunResult(
            task_id=self.example.id,
            target=self.example.target,
            impossible=self.example.impossible,
            success=bool(self.wrapper.success),
            reward=float(self.reward),
            steps=int(self.info.get("steps", self.action_count)),
            optimal_path_length=self.example.optimal_path_length,
            action_count=self.action_count,
            valid_action_count=self.valid_action_count,
            invalid_action_count=self.invalid_action_count,
            terminated=bool(self.terminated),
            truncated=bool(self.truncated),
            reason=str(self.info.get("reason", "")),
        )


def flatten_subplans(subplans: list[list[str]]) -> list[str]:
    return [action for subplan in subplans for action in subplan]


def run_oracle_episode(example, max_steps: int = 30) -> tuple[PlancraftRunResult, list[str]]:
    episode = PlancraftBenchEpisode(example, max_steps=max_steps)
    episode.reset()
    actions = flatten_subplans(episode.oracle_subplans())
    for action in actions:
        _text, _reward, terminated, truncated, _info = episode.step(action)
        if terminated or truncated:
            break
    return episode.result(), actions


def run_impossible_episode(example, max_steps: int = 30) -> tuple[PlancraftRunResult, list[str]]:
    episode = PlancraftBenchEpisode(example, max_steps=max_steps)
    episode.reset()
    action = "impossible: cannot craft target from the available inventory"
    episode.step(action)
    return episode.result(), [action]
