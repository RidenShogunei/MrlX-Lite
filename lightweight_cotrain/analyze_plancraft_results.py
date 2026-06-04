"""Evaluate simple Plancraft policies and write benchmark metrics."""

import argparse
import json
from pathlib import Path

from plancraft_environment import load_examples, run_impossible_episode, run_oracle_episode


def avg(rows, key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def write_summary(path: Path, rows: list[dict], policy: str):
    lines = ["# Plancraft Evaluation", ""]
    lines.append(f"policy = `{policy}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| tasks | {len(rows)} |")
    lines.append(f"| success_rate | {avg(rows, 'success'):.3f} |")
    lines.append(f"| reward | {avg(rows, 'reward'):.3f} |")
    lines.append(f"| efficiency | {avg(rows, 'efficiency'):.3f} |")
    lines.append(f"| avg_steps | {avg(rows, 'steps'):.3f} |")
    lines.append(f"| invalid_action_rate | {avg(rows, 'invalid_action_rate'):.3f} |")
    lines.append("")
    lines.append("## By Complexity")
    lines.append("")
    lines.append("| complexity | tasks | success_rate | reward | efficiency | avg_steps |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for complexity in sorted(set(row["complexity_split"] for row in rows)):
        subset = [row for row in rows if row["complexity_split"] == complexity]
        lines.append(
            f"| {complexity} | {len(subset)} | {avg(subset, 'success'):.3f} | "
            f"{avg(subset, 'reward'):.3f} | {avg(subset, 'efficiency'):.3f} | "
            f"{avg(subset, 'steps'):.3f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Plancraft benchmark policies.")
    parser.add_argument("--split", default="val.small.easy")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--policy", choices=["oracle", "impossible"], default="oracle")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--out-dir", default="./plancraft_eval_oracle")
    return parser.parse_args()


def main():
    args = parse_args()
    examples = load_examples(args.split, offset=args.offset, limit=args.tasks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for example in examples:
        if args.policy == "oracle":
            result, actions = run_oracle_episode(example, max_steps=args.max_steps)
        else:
            result, actions = run_impossible_episode(example, max_steps=args.max_steps)
        row = {
            **result.__dict__,
            "success": 1.0 if result.success else 0.0,
            "terminated": 1.0 if result.terminated else 0.0,
            "truncated": 1.0 if result.truncated else 0.0,
            "efficiency": result.efficiency,
            "invalid_action_rate": result.invalid_action_count / max(result.action_count, 1),
            "complexity_split": example.complexity_split,
            "complexity": example.complexity,
            "actions": actions,
        }
        rows.append(row)
        print(
            f"[plancraft:{args.policy}] id={example.id} target={example.target} "
            f"success={result.success} steps={result.steps} optimal={result.optimal_path_length}",
            flush=True,
        )

    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_summary(out_dir / "summary.md", rows, args.policy)
    print(f"[plancraft:{args.policy}] wrote {out_dir / 'results.jsonl'}")
    print(f"[plancraft:{args.policy}] wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
