"""
完整分析报告：五大实验组 × 多阶段对比
"""
import os, sys, re, json, torch
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from math_environment import MathEnvironment, MathReward, MathTask

_builtin_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _builtin_print(*args, **kwargs)


BASE_MODEL = "./models/qwen/Qwen2___5-1___5B-Instruct"


class CheckpointEntry:
    """一个待评估的 checkpoint"""
    def __init__(self, name: str, lora_path: str, experiment: str,
                 prompt_mode: str = "cn", stage_label: str = ""):
        self.name = name
        self.lora_path = lora_path
        self.experiment = experiment
        self.prompt_mode = prompt_mode  # "cn" | "en" | "en_suffix"
        self.stage_label = stage_label


def define_checkpoints() -> List[CheckpointEntry]:
    """定义所有要评估的 checkpoint"""
    entries = []

    # ===== 基线 0: 纯基础模型 =====
    entries.append(CheckpointEntry(
        "0-基础模型(无训练)", "", "Baseline", "cn", "none"
    ))

    # ===== 基线 1: SFT 格式训练 =====
    entries.append(CheckpointEntry(
        "1-SFT格式训练(Main)", "./sft_checkpoints/main_agent/main",
        "Baseline", "cn", "SFT"
    ))
    entries.append(CheckpointEntry(
        "1-SFT格式训练(Sub)", "./sft_checkpoints/sub_agent/sub",
        "Baseline", "cn", "SFT"
    ))

    # ===== 实验 A: SFT + 平滑奖励 GRPO (v4) =====
    for step in [5, 10, 15, 20]:
        lp = f"./sft_grpo_v4/main_step_{step}/main"
        if os.path.exists(lp):
            entries.append(CheckpointEntry(
                f"A-GRPO平滑step{step}", lp,
                "A-平滑奖励GRPO", "cn", f"step_{step}"
            ))

    # ===== 实验 B: MrlX 二进制奖励 (从SFT初始) =====
    for step in [10, 20]:
        lp = f"./sft_grpo_v3/main_step_{step}/main"
        if os.path.exists(lp):
            entries.append(CheckpointEntry(
                f"B-MrlX二值step{step}(SFT初始)", lp,
                "B-MrlX从SFT", "cn", f"step_{step}"
            ))

    # ===== 实验 C: MrlX 二进制奖励 (从基础模型初始) =====
    for step in [5, 10]:
        lp = f"./grpo_mrlx_v2/main_step_{step}/main"
        if os.path.exists(lp):
            entries.append(CheckpointEntry(
                f"C-MrlX二值step{step}(基础)", lp,
                "C-MrlX从零", "en_suffix", f"step_{step}"
            ))

    # ===== 实验 D: 原始 CoTrain (COT format) =====
    lp = "./cotrain_checkpoints_math_15b_v2/lora_main_step_20/main"
    if os.path.exists(lp):
        entries.append(CheckpointEntry(
            f"D-原始协同step20", lp,
            "D-原始协同训练", "cn", "step_20"
        ))

    return entries


class ModelLoader:
    def __init__(self):
        self.cached_model = None
        self.cached_tokenizer = None
        self.cached_lora_path = None

    def load(self, lora_path: str):
        if self.cached_model is not None and self.cached_lora_path == lora_path:
            return self.cached_model, self.cached_tokenizer

        if self.cached_model is not None:
            del self.cached_model
            torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, trust_remote_code=True,
            device_map="cuda:0", low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if lora_path and os.path.exists(lora_path):
            model = PeftModel.from_pretrained(model, lora_path, adapter_name="default")

        self.cached_model = model
        self.cached_tokenizer = tokenizer
        self.cached_lora_path = lora_path
        return model, tokenizer


def build_prompt(tokenizer, task: MathTask, mode: str) -> str:
    if mode == "cn":
        system = (
            "你是数学解题器，按格式回答：\n"
            "<thinking>思考过程</thinking>\n"
            "[tool_call]计算内容[/tool_call]\n"
            "<result>数字答案</result>"
        )
        user = f"问题: {task.question}"
    elif mode == "en":
        system = (
            "You are a math solver. Output ONLY in this format:\n"
            "<thinking>analysis</thinking>\n"
            "[tool_call]computation[/tool_call]\n"
            "<result>number</result>"
        )
        user = f"Question: {task.question}"
    elif mode == "en_suffix":
        system = (
            "YOU ARE A MATH SOLVER. OUTPUT ONLY:\n"
            "<thinking>analysis</thinking>\n"
            "[tool_call]computation[/tool_call]\n"
            "<result>number</result>\n\n"
            "Example: <thinking>sum</thinking>[tool_call]82+15[/tool_call]<result>97</result>"
        )
        user = f"Solve: {task.question}"

    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    if mode == "en_suffix":
        prompt += "<thinking>"
    return prompt


def generate_one(model, tokenizer, prompt: str, max_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=0.8, top_p=0.95, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    plen = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
    raw = re.sub(r'<\|im_start\|>.*?\n?', '', raw)
    raw = re.sub(r'<\|im_end\|>', '', raw).strip()
    return raw


def analyze_response(response: str, task: MathTask) -> Dict:
    last = response.rfind("</result>")
    if last >= 0:
        clean = response[:last + len("</result>")]
    else:
        clean = response

    is_valid = MathReward._validate_main_format(clean)
    reward = MathReward.compute_main_reward(task, clean, [])

    has_thinking = "<thinking>" in response and "</thinking>" in response
    has_tool = bool(re.search(r'\[tool_call\].*?\[/tool_call\]', response, re.DOTALL))
    has_result = "<result>" in response and "</result>" in response

    pred = MathEnvironment.extract_number(response)
    correct = pred is not None and MathEnvironment.check_answer(pred, task.answer)

    return {
        "state_machine_valid": is_valid,
        "has_thinking": has_thinking,
        "has_tool_call": has_tool,
        "has_result": has_result,
        "reward": reward,
        "predicted": pred,
        "correct": correct,
        "length": len(response),
        "response_snippet": response[:80],
    }


def evaluate_checkpoint(entry: CheckpointEntry, test_tasks: List[MathTask],
                        loader: ModelLoader, n_samples: int = 8) -> Dict:
    print(f"\n  📊 [{entry.name}] ...", end="")
    model, tokenizer = loader.load(entry.lora_path)
    all_results = []

    for task in test_tasks:
        prompt = build_prompt(tokenizer, task, entry.prompt_mode)
        task_results = []
        for _ in range(n_samples):
            resp = generate_one(model, tokenizer, prompt)
            ar = analyze_response(resp, task)
            task_results.append(ar)
        all_results.append(task_results)

    flat = [r for tr in all_results for r in tr]
    n_tasks = len(test_tasks)

    # 基础统计
    total = len(flat)
    state_valid = sum(1 for r in flat if r["state_machine_valid"]) / total
    thinking_rate = sum(1 for r in flat if r["has_thinking"]) / total
    tool_rate = sum(1 for r in flat if r["has_tool_call"]) / total
    result_rate = sum(1 for r in flat if r["has_result"]) / total

    correct_rate = sum(1 for r in flat if r["correct"]) / total
    avg_reward = sum(r["reward"] for r in flat) / total
    avg_len = sum(r["length"] for r in flat) / total

    # Reward 分布
    reward_hist = defaultdict(int)
    for r in flat:
        rv = r["reward"]
        if rv <= 0.01:
            reward_hist["0.00"] += 1
        elif rv <= 0.15:
            reward_hist["0.10"] += 1
        elif rv < 1.0:
            reward_hist["0.10<r<1.0"] += 1
        else:
            reward_hist["1.00"] += 1

    # Best-of-N
    best_correct = sum(
        1 for tr in all_results if any(r["correct"] for r in tr)
    )
    best_acc = best_correct / n_tasks
    best_reward_sum = sum(max(r["reward"] for r in tr) for tr in all_results)
    best_reward = best_reward_sum / n_tasks

    # 满分题数
    perfect = sum(1 for tr in all_results if any(r["reward"] >= 1.0 for r in tr))

    # Per-task 分解
    per_task = []
    for i, tr in enumerate(all_results):
        t_correct = sum(1 for r in tr if r["correct"])
        t_avg_r = sum(r["reward"] for r in tr) / len(tr)
        t_best_r = max(r["reward"] for r in tr)
        per_task.append({
            "idx": i,
            "question": test_tasks[i].question,
            "answer": test_tasks[i].answer,
            "correct_n": t_correct,
            "avg_reward": t_avg_r,
            "best_reward": t_best_r,
        })

    print(f" ✅ 状态机={state_valid:.1%} 正确率={correct_rate:.1%} "
          f"avgR={avg_reward:.3f} bestR={best_reward:.3f}")

    return {
        "name": entry.name,
        "experiment": entry.experiment,
        "stage_label": entry.stage_label,
        "n_samples": total,
        "n_tasks": n_tasks,
        "state_valid_rate": state_valid,
        "thinking_rate": thinking_rate,
        "tool_call_rate": tool_rate,
        "result_rate": result_rate,
        "correct_rate": correct_rate,
        "best_of_n_acc": best_acc,
        "avg_reward": avg_reward,
        "best_reward": best_reward,
        "perfect_tasks": perfect,
        "avg_length": avg_len,
        "reward_hist": dict(reward_hist),
        "per_task": per_task,
        "all_flat": flat,
    }


def print_header(title: str, width: int = 78):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def print_summary_table(all_eval: List[Dict]):
    print_header("📈 汇总对比表")
    hdr = (f"  {'实验/阶段':<28} {'状态机':>7} {'thinking':>9} {'tool':>7} "
           f"{'result':>8} {'正确率':>7} {'BestN':>6} {'平均R':>7} {'满分题':>6}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for er in all_eval:
        n = er["name"]
        print(f"  {n:<28} {er['state_valid_rate']:>6.1%} {er['thinking_rate']:>8.1%} "
              f"{er['tool_call_rate']:>6.1%} {er['result_rate']:>7.1%} "
              f"{er['correct_rate']:>6.1%} {er['best_of_n_acc']:>5.1%} "
              f"{er['avg_reward']:>6.3f} {er['perfect_tasks']:>5}/{er['n_tasks']}")


def print_reward_distribution(all_eval: List[Dict]):
    print_header("📊 奖励分布变化")
    for er in all_eval:
        hist = er["reward_hist"]
        total = er["n_samples"]
        parts = []
        for key in ["0.00", "0.10", "0.10<r<1.0", "1.00"]:
            cnt = hist.get(key, 0)
            pct = cnt / total * 100
            bar = "█" * int(pct / 2)
            parts.append(f"{key}:{cnt:>3}({pct:4.1f}%){bar}")
        print(f"  {er['name']:<30} {' | '.join(parts)}")


def print_correctness_trend(all_eval: List[Dict]):
    print_header("📈 准确率 & 奖励趋势")
    for metric, label in [("correct_rate", "答案正确率"), ("avg_reward", "平均MrlX奖励"),
                           ("state_valid_rate", "状态机格式通过率")]:
        print(f"\n  {label}:")
        max_v = max(er[metric] for er in all_eval) if all_eval else 1
        for er in all_eval:
            v = er[metric]
            bar = "█" * int(v / max(max_v, 0.01) * 40)
            print(f"    {er['name']:<30} {v:.3f}  {bar}")


def print_per_task_breakdown(all_eval: List[Dict], test_tasks: List[MathTask]):
    """展示每道题在各 checkpoint 下的表现"""
    print_header("📋 逐题对比 (Best-of-N 奖励)")

    # Header
    row_hdr = "  " + "题目".ljust(40) + " "
    for er in all_eval:
        row_hdr += f"{er['stage_label']:>9} "
    print(row_hdr)
    print("  " + "─" * (len(row_hdr) - 2))

    for i, task in enumerate(test_tasks):
        row = f"  [{i+1}] {task.question[:37].ljust(37)} "
        for er in all_eval:
            if i < len(er["per_task"]):
                br = er["per_task"][i]["best_reward"]
                sym = "🟢" if br >= 1.0 else ("🟡" if br >= 0.1 else "🔴")
                row += f"{sym}{br:.2f}   "
            else:
                row += "  N/A    "
        print(row)


def print_improvement_summary(all_eval: List[Dict]):
    print_header("🏆 关键发现")

    if len(all_eval) < 2:
        return

    baseline = all_eval[0]  # 基础模型
    best_by_acc = max(all_eval, key=lambda x: x["correct_rate"])
    best_by_reward = max(all_eval, key=lambda x: x["avg_reward"])
    best_by_state = max(all_eval, key=lambda x: x["state_valid_rate"])

    print(f"\n  最佳准确率: {best_by_acc['name']} ({best_by_acc['correct_rate']:.1%})")
    print(f"  最佳奖励:   {best_by_reward['name']} (R={best_by_reward['avg_reward']:.3f})")
    print(f"  最佳格式:   {best_by_state['name']} (状态机={best_by_state['state_valid_rate']:.1%})")

    # 找到"格式好+准确率高"的 checkpoint
    pareto = []
    for er in all_eval:
        if er["state_valid_rate"] > 0.95 and er["correct_rate"] > 0.7:
            pareto.append(er)
    if pareto:
        print(f"\n  ✅ 格式+准确率双优 ({len(pareto)} 个):")
        for p in pareto:
            print(f"     • {p['name']}: 格式={p['state_valid_rate']:.1%} 正确率={p['correct_rate']:.1%} "
                  f"BestN={p['best_of_n_acc']:.1%} 满分={p['perfect_tasks']}/{p['n_tasks']}")
    else:
        print(f"\n  ⚠️ 无checkpoint同时满足格式>95%和正确率>70%")

    # 最优推荐
    print(f"\n  💡 推荐 checkpoint 排名:")
    scored = []
    for er in all_eval:
        # 综合评分 = 正确率 × 0.5 + 最佳奖励 × 0.3 + 状态机通过率 × 0.2
        score = (er["correct_rate"] * 0.5 + er["best_reward"] * 0.3 +
                 er["state_valid_rate"] * 0.2)
        scored.append((er["name"], score, er["avg_reward"], er["correct_rate"]))
    scored.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, score, r, acc) in enumerate(scored[:5]):
        print(f"    {rank+1}. {name}  (综合={score:.3f}, 奖励={r:.3f}, 准确率={acc:.1%})")


def print_experiment_comparison(all_eval: List[Dict]):
    """按实验组对比"""
    print_header("🔬 实验组间对比")

    groups = defaultdict(list)
    for er in all_eval:
        groups[er["experiment"]].append(er)

    for exp_name, members in groups.items():
        if len(members) < 2:
            continue
        print(f"\n  [{exp_name}]")
        # 头尾对比
        first = members[0]
        last = members[-1]
        print(f"    开始 → 结束: 正确率 {first['correct_rate']:.1%} → {last['correct_rate']:.1%}  "
              f"奖励 {first['avg_reward']:.3f} → {last['avg_reward']:.3f}  "
              f"状态机 {first['state_valid_rate']:.1%} → {last['state_valid_rate']:.1%}")

        # 最佳
        best_in_group = max(members, key=lambda x: x["avg_reward"])
        print(f"    组内最佳: {best_in_group['stage_label']} (奖励={best_in_group['avg_reward']:.3f}, "
              f"正确率={best_in_group['correct_rate']:.1%})")


def main():
    print("=" * 78)
    print("  🔬 MrlX-GRPO 完整实验分析报告")
    print("=" * 78)
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    print(f"  基础模型: Qwen2.5-1.5B-Instruct")
    print(f"  每checkpoint采样: 8次/题")

    # 准备测试集
    env = MathEnvironment(seed=99)
    test_tasks = env.tasks[-20:]
    print(f"\n  测试题: {len(test_tasks)} 道 (独立于训练集)")

    # 定义所有 checkpoint
    entries = define_checkpoints()
    print(f"  待评估 checkpoint: {len(entries)} 个\n")

    # 逐个评估
    loader = ModelLoader()
    all_eval = []

    for i, entry in enumerate(entries):
        try:
            er = evaluate_checkpoint(entry, test_tasks, loader, n_samples=8)
            all_eval.append(er)
        except Exception as e:
            print(f"  ❌ 失败: {e}")

    # ===== 输出报告 =====
    print_summary_table(all_eval)
    print_correctness_trend(all_eval)
    print_reward_distribution(all_eval)
    print_per_task_breakdown(all_eval, test_tasks)
    print_experiment_comparison(all_eval)
    print_improvement_summary(all_eval)

    # 保存 JSON
    report_path = "analysis_report.json"
    json_data = []
    for er in all_eval:
        d = {k: v for k, v in er.items() if k not in ("all_flat", "per_task")}
        json_data.append(d)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 报告已保存: {report_path}")

    print(f"\n{'='*78}")
    print(f"  ✅ 分析完成！")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
