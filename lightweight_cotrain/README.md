# 轻量级 MrlX 协同训练实验

这是一个基于 MrlX 思路的本地轻量复刻，用 Qwen2.5 + LoRA 验证 Main Agent / Sub Agent 在数学任务上的格式约束、奖励设计和协同训练稳定性。

当前代码更偏研究原型，不是完整框架。仓库只跟踪源码、SFT 数据和分析文档；模型与 checkpoint 目录被 `.gitignore` 排除。

## 项目结构

| 文件 | 作用 |
| --- | --- |
| `math_environment.py` | 生成数学任务、抽取答案、检查格式、计算 Main/Sub 奖励 |
| `generate_sft_data.py` | 生成 Main/Sub 格式化 SFT 数据到 `sft_data.jsonl` |
| `sft_trainer.py` | 训练 Main/Sub LoRA 格式适配器，默认输出到 `sft_checkpoints/` |
| `grpo_v4.py` | 从 SFT adapter 出发，做 Best-of-N + 高奖励样本更新 |
| `cotrain_system_main_sub.py` | 更完整的 Main -> Sub 协同训练实验脚本 |
| `analyze_results.py` | 对 SFT/GRPO checkpoint 做基础评估 |
| `full_analysis.py` | 自动发现本地 LoRA checkpoint 并做多阶段评估 |
| `ANALYSIS_REPORT.md` | 既有实验结论与问题分析 |

## 环境

```bash
pip install -r requirements.txt
```

如启用 4bit 量化，还需要安装与本机 CUDA 匹配的 `bitsandbytes`。

## 模型路径

主要脚本默认使用：

```text
./models/qwen/Qwen2___5-1___5B-Instruct
```

旧实验目录里也有 0.5B checkpoint。运行脚本前请确认对应模型已经下载到 `models/` 下，或在脚本里的 `CoTrainConfig.base_model` 中改成实际路径。

## 推荐流程

1. 生成 SFT 数据：

```bash
python generate_sft_data.py
```

2. 训练 Main/Sub SFT adapter：

```bash
python sft_trainer.py
```

3. 从 SFT adapter 继续跑 GRPO v4：

```bash
python grpo_v4.py
```

4. 评估本地已有 checkpoint：

```bash
python full_analysis.py
```

`full_analysis.py` 会自动跳过不存在的历史目录，并扫描当前存在的 LoRA adapter。

## 当前实验结论摘要

- 平滑奖励能提升答案正确率，但容易让模型学会丢掉 XML/工具调用格式。
- MrlX 式二值奖励更强调格式，但在小模型、小 batch、无强 KL 约束时信号稀疏，训练容易崩。
- SFT 数据与 GRPO prompt 的格式必须完全一致，否则会出现“有答案、没格式、奖励全零”的情况。
- 最需要继续改进的是奖励函数、KL/参考模型约束、SFT 数据覆盖和 checkpoint 评估脚本的一致性。

