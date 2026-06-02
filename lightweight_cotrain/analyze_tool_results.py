"""Evaluate Main adapters on the mini tool-use environment."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from generate_tool_sft_data import MAIN_SYSTEM
from tool_environment import ToolEnvironment, ToolTask


def load_model(base_model: str, lora_path: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path, adapter_name="default")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, task: ToolTask) -> str:
    user = f"products 表：\n{ToolEnvironment.render_table(task.db_rows)}\n\n问题：{task.question}"
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": MAIN_SYSTEM},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_one(model, tokenizer, prompt: str, device: str, max_tokens: int, use_prefix: bool) -> str:
    prefix = "<thinking>" if use_prefix else ""
    inputs = tokenizer(prompt + prefix, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    response = prefix + generated
    end = response.find("</result>")
    if end >= 0:
        response = response[:end + len("</result>")]
    return response.strip()


def evaluate(model, tokenizer, tasks, device: str, samples: int, max_tokens: int, use_prefix: bool):
    raw_rewards, executed_rewards, tool_valids = [], [], []
    best_executed = 0.0

    for task in tasks:
        prompt = build_prompt(tokenizer, task)
        task_best = 0.0
        for _ in range(samples):
            raw = generate_one(model, tokenizer, prompt, device, max_tokens, use_prefix)
            ok, _ = ToolEnvironment.execute_tool(task, raw)
            fixed = ToolEnvironment.canonicalize_response(task, raw)
            raw_reward = ToolEnvironment.reward(task, raw)
            executed_reward = ToolEnvironment.reward(task, fixed)

            raw_rewards.append(raw_reward)
            executed_rewards.append(executed_reward)
            tool_valids.append(1.0 if ok else 0.0)
            task_best = max(task_best, executed_reward)
        best_executed += task_best

    total = max(len(raw_rewards), 1)
    return {
        "raw_reward": sum(raw_rewards) / total,
        "executed_reward": sum(executed_rewards) / total,
        "tool_valid": sum(tool_valids) / total,
        "best_executed": best_executed / max(len(tasks), 1),
        "samples": total,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tool-use SFT/GRPO adapters.")
    parser.add_argument("--base-model", default="./models/qwen/Qwen2___5-1___5B-Instruct")
    parser.add_argument("--lora", default="./tool_sft_checkpoints/main_agent/main")
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--raw", action="store_true", help="Do not force the <thinking> prefix.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _, _, test_tasks = ToolEnvironment.split(0, 0, args.tasks, seed=args.seed)
    model, tokenizer = load_model(args.base_model, args.lora, device)
    metrics = evaluate(
        model,
        tokenizer,
        test_tasks,
        device=device,
        samples=args.samples,
        max_tokens=args.max_tokens,
        use_prefix=not args.raw,
    )
    print("[tool-eval]")
    print(f"lora={args.lora or '(base)'}")
    print(f"tasks={args.tasks} samples={args.samples} raw_mode={args.raw}")
    print(f"tool_valid={metrics['tool_valid']:.3f}")
    print(f"raw_reward={metrics['raw_reward']:.3f}")
    print(f"executed_reward={metrics['executed_reward']:.3f}")
    print(f"best_executed={metrics['best_executed']:.3f}")


if __name__ == "__main__":
    main()
