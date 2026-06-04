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

STRUCTURED_SUB_SYSTEM = (
    "You are a Plancraft sub agent. Inspect the current objective, inventory, and action history.\n"
    "Give structured guidance for the main agent.\n"
    "Output exactly this format:\n"
    "<subgoal>local crafting goal</subgoal>\n"
    "<reason>brief reason based on the current state</reason>\n"
    "<action>one recommended low-level action</action>\n"
    "Valid action formats inside <action> are:\n"
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


def history_text(history: list[tuple[str, str]], history_steps: int = 3) -> str:
    if not history:
        return "No actions yet."
    lines = []
    start_step = max(len(history) - history_steps, 0)
    for idx, (action, observation) in enumerate(history[start_step:], start_step + 1):
        lines.append(f"Step {idx} action: {action}")
        lines.append(f"Step {idx} observation: {observation}")
    return "\n".join(lines)


def sub_user(observation: str, history: list[tuple[str, str]], history_steps: int = 3) -> str:
    return f"Current observation:\n{observation}\n\nHistory:\n{history_text(history, history_steps)}"


def main_user(observation: str, history: list[tuple[str, str]], sub_advice: str, history_steps: int = 3) -> str:
    return (
        f"Current observation:\n{observation}\n\n"
        f"History:\n{history_text(history, history_steps)}\n\n"
        f"Sub agent advice:\n{sub_advice}"
    )


def structured_sub_advice(example, action: str) -> str:
    if action.startswith("impossible:"):
        return (
            f"<subgoal>decide whether {example.target} is craftable</subgoal>\n"
            "<reason>The oracle marks this task as impossible from the available inventory.</reason>\n"
            f"<action>{action}</action>"
        )
    return (
        f"<subgoal>make progress toward crafting {example.target}</subgoal>\n"
        f"<reason>The next oracle step for the current state is: {action}</reason>\n"
        f"<action>{action}</action>"
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


def build_samples_for_example(example, max_steps: int, structured_sub: bool = False, history_steps: int = 3) -> list[dict]:
    episode = PlancraftBenchEpisode(example, max_steps=max_steps)
    observation = episode.reset()
    history: list[tuple[str, str]] = []
    samples = []
    sub_system = STRUCTURED_SUB_SYSTEM if structured_sub else SUB_SYSTEM
    sub_stage = "plancraft_sub_structured" if structured_sub else "plancraft_sub_action"

    if example.impossible:
        action = "impossible: cannot craft target from the available inventory"
        sub_advice = structured_sub_advice(example, action) if structured_sub else action
        samples.append(
            make_sample(sub_system, sub_user(observation, history, history_steps), sub_advice, "sub", sub_stage, example)
        )
        samples.append(
            make_sample(
                MAIN_SYSTEM,
                main_user(observation, history, sub_advice, history_steps),
                action,
                "main",
                "plancraft_main_action",
                example,
            )
        )
        return samples

    actions = [action for subplan in episode.oracle_subplans() for action in subplan]
    for action in actions[:max_steps]:
        sub_advice = structured_sub_advice(example, action) if structured_sub else action
        samples.append(
            make_sample(sub_system, sub_user(observation, history, history_steps), sub_advice, "sub", sub_stage, example)
        )
        samples.append(
            make_sample(
                MAIN_SYSTEM,
                main_user(observation, history, sub_advice, history_steps),
                action,
                "main",
                "plancraft_main_action",
                example,
            )
        )
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
    parser.add_argument("--history-steps", type=int, default=3)
    parser.add_argument("--include-impossible", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--structured-sub", action=argparse.BooleanOptionalAction, default=False)
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
        samples.extend(
            build_samples_for_example(
                example,
                args.max_steps,
                structured_sub=args.structured_sub,
                history_steps=args.history_steps,
            )
        )

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[plancraft-mas-sft] wrote {len(samples)} samples to {out}")
    print(f"[plancraft-mas-sft] main={sum(1 for s in samples if s['category'] == 'main')}")
    print(f"[plancraft-mas-sft] sub={sum(1 for s in samples if s['category'] == 'sub')}")
    print(f"[plancraft-mas-sft] structured_sub={args.structured_sub}")
    print(f"[plancraft-mas-sft] history_steps={args.history_steps}")
    print(f"[plancraft-mas-sft] skipped_impossible={skipped_impossible}")


if __name__ == "__main__":
    main()
