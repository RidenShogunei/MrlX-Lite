"""Generate Main/Sub SFT JSONL from Plancraft oracle subplans."""

import argparse
import json
from pathlib import Path

from plancraft_environment import PlancraftBenchEpisode, load_examples


SUB_SYSTEM = (
    "You are a Plancraft sub agent. Inspect the current objective, inventory, and action history.\n"
    "Suggest exactly one next low-level action that may help craft the target.\n"
    "Valid action formats are:\n"
    "move: from [Source] to [Target] with quantity N\n"
    "smelt: from [Source] to [Target] with quantity N\n"
    "impossible: short reason\n"
    "Slots include [A1]-[C3], [I1]-[I36], and output slot [0]."
)

MAIN_SYSTEM = (
    "You are a Plancraft main agent. Use the sub agent advice, but output exactly one executable action.\n"
    "Valid action formats are:\n"
    "move: from [Source] to [Target] with quantity N\n"
    "smelt: from [Source] to [Target] with quantity N\n"
    "impossible: short reason\n"
    "Do not output explanations after the action."
)


def history_text(history: list[tuple[str, str]]) -> str:
    if not history:
        return "No actions yet."
    lines = []
    for idx, (action, observation) in enumerate(history, 1):
        lines.append(f"Step {idx} action: {action}")
        lines.append(f"Step {idx} observation: {observation}")
    return "\n".join(lines[-12:])


def sub_user(observation: str, history: list[tuple[str, str]]) -> str:
    return f"Current observation:\n{observation}\n\nHistory:\n{history_text(history)}"


def main_user(observation: str, history: list[tuple[str, str]], sub_advice: str) -> str:
    return (
        f"Current observation:\n{observation}\n\n"
        f"History:\n{history_text(history)}\n\n"
        f"Sub agent advice:\n{sub_advice}"
    )


def make_sample(system: str, user: str, assistant: str, category: str, stage: str, example) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "category": category,
        "stage": stage,
        "task_type": example.complexity_split,
        "task_id": example.id,
        "target": example.target,
        "impossible": example.impossible,
    }


def build_samples_for_example(example, max_steps: int) -> list[dict]:
    episode = PlancraftBenchEpisode(example, max_steps=max_steps)
    observation = episode.reset()
    history: list[tuple[str, str]] = []
    samples = []

    if example.impossible:
        action = "impossible: cannot craft target from the available inventory"
        sub_advice = action
        samples.append(make_sample(SUB_SYSTEM, sub_user(observation, history), sub_advice, "sub", "plancraft_sub_action", example))
        samples.append(make_sample(MAIN_SYSTEM, main_user(observation, history, sub_advice), action, "main", "plancraft_main_action", example))
        return samples

    actions = [action for subplan in episode.oracle_subplans() for action in subplan]
    for action in actions[:max_steps]:
        sub_advice = action
        samples.append(make_sample(SUB_SYSTEM, sub_user(observation, history), sub_advice, "sub", "plancraft_sub_action", example))
        samples.append(make_sample(MAIN_SYSTEM, main_user(observation, history, sub_advice), action, "main", "plancraft_main_action", example))
        observation, _reward, terminated, truncated, _info = episode.step(action)
        history.append((action, observation))
        if terminated or truncated:
            break
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Plancraft Main/Sub SFT data.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="plancraft_mas_sft_data.jsonl")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--include-impossible", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    examples = load_examples(args.split, offset=args.offset, limit=args.limit)
    samples = []
    skipped_impossible = 0
    for example in examples:
        if example.impossible and not args.include_impossible:
            skipped_impossible += 1
            continue
        samples.extend(build_samples_for_example(example, args.max_steps))

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[plancraft-mas-sft] wrote {len(samples)} samples to {out}")
    print(f"[plancraft-mas-sft] main={sum(1 for s in samples if s['category'] == 'main')}")
    print(f"[plancraft-mas-sft] sub={sum(1 for s in samples if s['category'] == 'sub')}")
    print(f"[plancraft-mas-sft] skipped_impossible={skipped_impossible}")


if __name__ == "__main__":
    main()
