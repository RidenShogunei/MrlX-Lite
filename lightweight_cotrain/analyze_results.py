"""
综合分析：SFT基线 vs GRPO各阶段效果对比
"""
import os, sys, re, torch
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from math_environment import MathEnvironment, MathReward, MathTask

_builtin_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _builtin_print(*args, **kwargs)


def load_checkpoint(base_model_path: str, lora_path: str) -> Tuple:
    """加载一个 checkpoint"""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, lora_path, adapter_name="default")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_prompt(tokenizer, task: MathTask, is_main: bool = True) -> str:
    if is_main:
        system = (
            "你是数学解题器，按格式回答：\n"
            "<thinking>思考过程</thinking>\n"
            "[tool_call]计算内容[/tool_call]\n"
            "<result>数字答案</result>"
        )
        user = f"问题: {task.question}"
    else:
        system = "执行计算，格式：<thinking>过程</thinking><result>数字</result>"
        user = f"计算: "
    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, prompt: str, max_tokens: int = 128, n_samples: int = 8) -> List[str]:
    responses = []
    for _ in range(n_samples):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=0.8, top_p=0.95,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        plen = inputs["input_ids"].shape[1]
        resp = tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
        responses.append(resp)
    return responses


def analyze_response(response: str, task: MathTask) -> Dict:
    """单条响应分析"""
    analysis = {}

    analysis["has_thinking"] = "<thinking>" in response and "</thinking>" in response
    analysis["has_tool_call"] = bool(re.search(r'\[tool_call\].*?\[/tool_call\]', response, re.DOTALL))
    analysis["has_result"] = "<result>" in response and "</result>" in response

    m = re.search(r"<result>\s*(-?\d+\.?\d*)\s*</result>", response)
    analysis["result_tag_value"] = m.group(1) if m else None

    pred = MathEnvironment.extract_number(response)
    analysis["predicted"] = pred
    analysis["correct"] = pred is not None and MathEnvironment.check_answer(pred, task.answer)

    analysis["reward"] = MathReward.compute_main_reward(task, response, [])
    analysis["length"] = len(response)

    return analysis


def evaluate_checkpoint(model, tokenizer, test_tasks: List[MathTask], name: str):
    """在一个 checkpoint 上全面评估"""
    print(f"\n{'='*70}")
    print(f"  📊 评估: {name}")
    print(f"{'='*70}")

    all_results = []
    for task in test_tasks:
        prompt = build_prompt(tokenizer, task, is_main=True)
        responses = generate(model, tokenizer, prompt, n_samples=8)

        task_results = []
        for resp in responses:
            r = analyze_response(resp, task)
            task_results.append(r)
        all_results.append(task_results)

    # 统计
    n_tasks = len(test_tasks)
    n_samples = len(all_results[0]) if all_results else 0
    flat = [r for tr in all_results for r in tr]

    thinking_rate = sum(1 for r in flat if r["has_thinking"]) / len(flat)
    tool_call_rate = sum(1 for r in flat if r["has_tool_call"]) / len(flat)
    result_rate = sum(1 for r in flat if r["has_result"]) / len(flat)
    correct_rate = sum(1 for r in flat if r["correct"]) / len(flat)
    avg_reward = sum(r["reward"] for r in flat) / len(flat)
    avg_length = sum(r["length"] for r in flat) / len(flat)

    # Best-of-N 准确率
    best_correct = 0
    best_avg_reward = 0
    for task_results in all_results:
        best_r = max(task_results, key=lambda r: r["reward"])
        if best_r["correct"]:
            best_correct += 1
        best_avg_reward += best_r["reward"]
    best_acc = best_correct / n_tasks
    best_r_mean = best_avg_reward / n_tasks

    # 奖励分布
    reward_bins = {"0.0": 0, "0.0-0.3": 0, "0.3-0.6": 0, "0.6-0.9": 0, "0.9-1.0": 0}
    for r in flat:
        rv = r["reward"]
        if rv <= 0.001:
            reward_bins["0.0"] += 1
        elif rv < 0.3:
            reward_bins["0.0-0.3"] += 1
        elif rv < 0.6:
            reward_bins["0.3-0.6"] += 1
        elif rv < 0.9:
            reward_bins["0.6-0.9"] += 1
        else:
            reward_bins["0.9-1.0"] += 1

    print(f"  样本数: {len(flat)} ({n_tasks}题 × {n_samples}次采样)")
    print(f"  格式: <thinking>={thinking_rate:.1%}  [tool_call]={tool_call_rate:.1%}  <result>={result_rate:.1%}")
    print(f"  答案正确率: {correct_rate:.1%}")
    print(f"  Best-of-{n_samples} 正确率: {best_acc:.1%}")
    print(f"  平均奖励: {avg_reward:.3f}  平均长度: {avg_length:.0f} chars")
    print(f"  奖励分布:")
    for bin_name, count in sorted(reward_bins.items()):
        bar = "█" * (count * 40 // len(flat))
        print(f"    {bin_name:>8}: {count:>3} [{bar}]")

    return {
        "name": name,
        "thinking_rate": thinking_rate,
        "tool_call_rate": tool_call_rate,
        "result_rate": result_rate,
        "correct_rate": correct_rate,
        "best_acc": best_acc,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "reward_bins": reward_bins,
        "all_flat": flat,
        "all_results": all_results,
    }


def show_sample_outputs(all_eval_results: List[Dict], test_tasks: List[MathTask]):
    """展示典型样本"""
    print(f"\n{'='*70}")
    print(f"  📝 样本对比")
    print(f"{'='*70}")

    # 挑一题展示各阶段
    task = test_tasks[0]
    print(f"\n  题目: {task.question}")
    print(f"  答案: {task.answer}")

    for eval_res in all_eval_results:
        name = eval_res["name"]
        task_results = eval_res["all_results"][0]  # 第一题
        best = max(task_results, key=lambda r: r["reward"])
        print(f"\n  --- {name} ---")
        print(f"  Reward: {best['reward']:.3f}")
        resp_text = best.get("response", "N/A") if "response" in best else "(省略)"
        if best["predicted"] is not None:
            status = "✅" if best["correct"] else "❌"
            print(f"  预测: {best['predicted']} {status}")
        print(f"  响应: {best.get('response_text', '(未存储完整文本)')[:200]}")


def print_summary_table(all_eval_results: List[Dict]):
    """打印汇总表"""
    print(f"\n{'='*70}")
    print(f"  📈 汇总对比")
    print(f"{'='*70}")
    print(f"  {'阶段':<18} {'格式':>8} {'tool_call':>10} {'<result>':>9} {'正确率':>8} {'Best-N':>8} {'平均奖励':>9}")
    print(f"  {'─'*18} {'─'*8} {'─'*10} {'─'*9} {'─'*8} {'─'*8} {'─'*9}")

    for er in all_eval_results:
        print(f"  {er['name']:<18} {er['thinking_rate']:>7.1%} {er['tool_call_rate']:>9.1%} "
              f"{er['result_rate']:>8.1%} {er['correct_rate']:>7.1%} {er['best_acc']:>7.1%} {er['avg_reward']:>8.3f}")


def main():
    print("=" * 70)
    print("  🔬 SFT + GRPO 综合分析")
    print("=" * 70)
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    base_model_path = "./models/qwen/Qwen2___5-1___5B-Instruct"

    # 测试题（不在训练集中的）
    env = MathEnvironment(seed=99)
    test_tasks = env.tasks[-10:]  # 最后10题
    print(f"\n[系统] 测试题数: {len(test_tasks)}")
    for i, t in enumerate(test_tasks):
        print(f"  [{i+1}] {t.question[:50]}... (ans={t.answer})")

    all_eval_results = []

    # --- 0: 纯基础模型（无 LoRA）---
    print(f"\n\n{'#'*70}")
    print(f"{'#':^70}")
    print(f"  {'阶段 0: 纯基础模型 (Qwen2.5-1.5B-Instruct)':^62}")
    print(f"{'#':^70}")
    print(f"{'#'*70}")
    try:
        model, tokenizer = load_checkpoint(base_model_path, "./sft_checkpoints/main_agent/main")
        # 重置 LoRA 权重为零（相当于不用）
        er = evaluate_checkpoint(model, tokenizer, test_tasks, "0-基础模型(无训练)")
        all_eval_results.append(er)
        del model; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")

    # --- 1: SFT 后（Main Agent）---
    print(f"\n\n{'#'*70}")
    print(f"  {'阶段 1: SFT 后（仅格式训练）':^62}")
    print(f"{'#'*70}")
    model, tokenizer = load_checkpoint(base_model_path, "./sft_checkpoints/main_agent/main")
    er = evaluate_checkpoint(model, tokenizer, test_tasks, "1-SFT格式化后")
    all_eval_results.append(er)
    del model; torch.cuda.empty_cache()

    # --- 2-5: GRPO 各阶段---
    for step in [5, 10, 15, 20]:
        lora_dir = f"./sft_grpo_v4/main_step_{step}/main"
        if not os.path.exists(lora_dir):
            print(f"\n  ⚠️ {lora_dir} 不存在，跳过")
            continue

        print(f"\n\n{'#'*70}")
        print(f"  {'阶段 2: GRPO step ' + str(step):^62}")
        print(f"{'#'*70}")
        model, tokenizer = load_checkpoint(base_model_path, lora_dir)
        er = evaluate_checkpoint(model, tokenizer, test_tasks, f"2-GRPO_step_{step}")
        all_eval_results.append(er)
        del model; torch.cuda.empty_cache()

    # 打印汇总
    print_summary_table(all_eval_results)

    # 键指标趋势
    print(f"\n{'='*70}")
    print(f"  📈 关键指标趋势")
    print(f"{'='*70}")
    for metric, label in [("avg_reward","平均奖励"), ("correct_rate","正确率"), ("best_acc","Best-N正确率")]:
        labels = [er["name"] for er in all_eval_results]
        values = [er[metric] for er in all_eval_results]
        print(f"  {label}:")
        for lb, v in zip(labels, values):
            bar_len = int(v * 40)
            print(f"    {lb:<22} {v:.3f}  {'█'*bar_len}")

    # 奖励分布变化
    print(f"\n{'='*70}")
    print(f"  📊 奖励分布变化")
    print(f"{'='*70}")
    for er in all_eval_results:
        bins = er["reward_bins"]
        total = sum(bins.values())
        zero_pct = bins["0.0"] / total * 100 if total > 0 else 0
        high_pct = (bins["0.6-0.9"] + bins["0.9-1.0"]) / total * 100 if total > 0 else 0
        print(f"  {er['name']:<22} 零奖励={zero_pct:5.1f}%  高奖励(>0.6)={high_pct:5.1f}%")

    print(f"\n{'='*70}")
    print(f"  ✅ 分析完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
