"""Evaluate existing Main/Sub LoRA agents on Plancraft text-only tasks."""

import argparse
import json
import random
from pathlib import Path

import torch

from analyze_hotpotqa_mas_results import build_prompt, load_model, set_adapter
from plancraft_environment import PlancraftBenchEpisode, load_examples
from plancraft_prompts import (
    MAIN_SYSTEM,
    STRUCTURED_SUB_SYSTEM,
    SUB_SYSTEM,
    history_text,
)


def truncate_generation(text: str, structured: bool = False) -> str:
    text = text.strip()
    if not text:
        return ""
    if structured:
        end = text.find("</action>")
        if end >= 0:
            return text[: end + len("</action>")].strip()
        return "\n".join(text.splitlines()[:3]).strip()
    return text.splitlines()[0].strip()


def generate_action(model, tokenizer, adapter: str, prompt: str, device: str, max_tokens: int, structured: bool = False):
    set_adapter(model, adapter)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return truncate_generation(text, structured=structured)


def run_mas_episode(model, tokenizer, example, device: str, max_steps: int, max_tokens: int, structured_sub: bool = False):
    episode = PlancraftBenchEpisode(example, max_steps=max_steps)
    observation = episode.reset()
    history = []
    trace = []
    sub_system = STRUCTURED_SUB_SYSTEM if structured_sub else SUB_SYSTEM
    for _step in range(max_steps):
        sub_prompt = build_prompt(
            tokenizer,
            sub_system,
            f"Current observation:\n{observation}\n\nHistory:\n{history_text(history)}",
        )
        sub_raw = generate_action(model, tokenizer, "sub", sub_prompt, device, max_tokens, structured=structured_sub)
        main_prompt = build_prompt(
            tokenizer,
            MAIN_SYSTEM,
            (
                f"Current observation:\n{observation}\n\n"
                f"History:\n{history_text(history)}\n\n"
                f"Sub agent advice:\n{sub_raw}"
            ),
        )
        main_raw = generate_action(model, tokenizer, "main", main_prompt, device, max_tokens)
        observation, reward, terminated, truncated, info = episode.step(main_raw)
        history.append((sub_raw, main_raw, observation))
        trace.append(
            {
                "sub_raw": sub_raw,
                "main_raw": main_raw,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )
        if terminated or truncated:
            break
    return episode.result(), trace


def avg(rows, key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Main/Sub agents on Plancraft.")
    parser.add_argument("--base-model", default="./models/qwen/Qwen2___5-1___5B-Instruct")
    parser.add_argument("--main-lora", default="./hotpotqa_mas_enhanced_mainonly_conservative_50x1/best/main")
    parser.add_argument("--sub-lora", default="./hotpotqa_mas_enhanced_mainonly_conservative_50x1/best/sub")
    parser.add_argument("--split", default="val.small.easy")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="./plancraft_eval_mas")
    parser.add_argument("--structured-sub", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    examples = load_examples(args.split, offset=args.offset, limit=args.tasks)
    model, tokenizer = load_model(args.base_model, args.main_lora, args.sub_lora, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for example in examples:
        result, trace = run_mas_episode(
            model,
            tokenizer,
            example,
            device,
            args.max_steps,
            args.max_tokens,
            structured_sub=args.structured_sub,
        )
        row = {
            **result.__dict__,
            "success": 1.0 if result.success else 0.0,
            "efficiency": result.efficiency,
            "invalid_action_rate": result.invalid_action_count / max(result.action_count, 1),
            "complexity_split": example.complexity_split,
            "complexity": example.complexity,
            "trace": trace,
        }
        rows.append(row)
        print(
            f"[plancraft:mas] id={example.id} target={example.target} "
            f"success={result.success} steps={result.steps} invalid={result.invalid_action_count}",
            flush=True,
        )

    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = ["# Plancraft MAS Evaluation", ""]
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| tasks | {len(rows)} |")
    lines.append(f"| success_rate | {avg(rows, 'success'):.3f} |")
    lines.append(f"| reward | {avg(rows, 'reward'):.3f} |")
    lines.append(f"| efficiency | {avg(rows, 'efficiency'):.3f} |")
    lines.append(f"| avg_steps | {avg(rows, 'steps'):.3f} |")
    lines.append(f"| invalid_action_rate | {avg(rows, 'invalid_action_rate'):.3f} |")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[plancraft:mas] wrote {out_dir / 'results.jsonl'}")
    print(f"[plancraft:mas] wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
