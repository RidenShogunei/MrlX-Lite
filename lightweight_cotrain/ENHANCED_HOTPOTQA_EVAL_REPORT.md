# Enhanced HotpotQA Evaluation Report

## Setup

正式评估使用 enhanced HotpotQA：

```text
val_jsonl = .\hotpotqa_data_enhanced\val.jsonl
offsets = 0, 20, 40, 60, 80
tasks_per_offset = 20
samples = 2
seed = 123
docs_per_task = 30
```

## Direct Main

结果文件：

```text
.\hotpotqa_direct_eval_enhanced_offsets\summary.md
```

Task-weighted average：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Direct Main SFT | 0.301 | 0.455 | 0.402 | 0.364 | 0.449 |
| Direct Main GRPO 150x3 | 0.372 | 0.515 | 0.463 | 0.410 | 0.491 |

Direct GRPO 在 enhanced 环境上仍有迁移收益：

```text
answer_f1 +0.071
reward    +0.061
```

## MAS

结果文件：

```text
.\hotpotqa_mas_eval_enhanced_offsets\summary.md
```

Task-weighted average：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| MAS Stage 2 | 0.313 | 0.372 | 0.394 | 0.400 | 0.462 |
| MAS Joint 30x1 | 0.286 | 0.372 | 0.375 | 0.366 | 0.436 |

MAS 内部结论：

```text
Stage 2 > Joint。
```

## Direct vs MAS

核心对照：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Direct Main GRPO 150x3 | 0.372 | 0.515 | 0.463 | 0.410 | 0.491 |
| MAS Stage 2 | 0.313 | 0.372 | 0.394 | 0.400 | 0.462 |
| MAS Joint 30x1 | 0.286 | 0.372 | 0.375 | 0.366 | 0.436 |

正式结论：

```text
Enhanced 30-doc 环境确实让任务更难，但当前旧 MAS checkpoint 仍没有超过 direct Main GRPO。
```

相比原环境，direct 和 MAS 的 reward gap 略有缩小：

```text
原环境:
  Direct GRPO reward ~= 0.510
  MAS Stage2 reward  ~= 0.434
  gap ~= 0.076

Enhanced:
  Direct GRPO reward = 0.463
  MAS Stage2 reward  = 0.394
  gap = 0.069
```

方向是对的，但 30 docs 还不足以自然产生 MAS 优势。

## Enhanced Sub Oracle

结果文件：

```text
.\hotpotqa_sub_eval_enhanced_offsets\summary.md
```

Task-weighted average：

| Sub | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Preference / Stage2 Sub | 0.378 | 0.285 | 0.378 | 0.375 | 0.415 | 0.356 | 0.430 |
| Joint Sub | 0.355 | 0.275 | 0.357 | 0.364 | 0.385 | 0.332 | 0.410 |

Sub oracle 结论：

```text
30-doc enhanced 环境下，Sub retrieval 下降明显。
Preference/Stage2 Sub 仍优于 Joint Sub。
```

这定位了 MAS 在 enhanced 上输给 direct 的主要原因：

```text
Sub 在更多 distractors 下没有足够强的 retrieval 能力；
Joint 会轻微损伤 Sub retrieval；
Main 即便能整合，也拿不到足够好的 Sub evidence。
```

## Next Step

现在不应该直接继续用旧 checkpoint 做 joint。下一步应该在 enhanced 环境上重新训练 Sub retrieval：

```text
1. 用 enhanced train 生成新的 preference pairs。
2. 从现有 Preference Sub 出发，训练 enhanced preference Sub。
3. 用 enhanced Sub oracle 验证 support_read_recall 是否超过 0.378。
4. 冻结 enhanced Sub，再训练 Main。
```

推荐下一条实验：

```bash
python train_hotpotqa_sub_preferences.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --tasks 300 ^
  --max-pairs 900 ^
  --epochs 1 ^
  --sub-lora .\hotpotqa_sub_pref_100x250\sub ^
  --save-dir .\hotpotqa_sub_pref_enhanced_300x900\sub ^
  --lr 5e-6 ^
  --beta 2.0 ^
  --sft-weight 0.05
```

成功标准：

```text
Enhanced Sub oracle:
support_read_recall > 0.378
reward > 0.375
```

## Enhanced Sub Preference Training

按上述路线，使用 enhanced train 重新训练 Sub retrieval：

```bash
python train_hotpotqa_sub_preferences.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --tasks 300 ^
  --max-pairs 900 ^
  --epochs 1 ^
  --sub-lora .\hotpotqa_sub_pref_100x250\sub ^
  --save-dir .\hotpotqa_sub_pref_enhanced_300x900\sub ^
  --lr 5e-6 ^
  --beta 2.0 ^
  --sft-weight 0.05 ^
  --max-length 1536
```

训练结果：

```text
tasks = 300
pairs = 900
epoch = 1
loss = 0.0037
margin = 5.3887
```

## Enhanced Sub Oracle After Preference

评估：

```bash
python analyze_hotpotqa_sub_oracle.py ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --offset 0 ^
  --tasks 100 ^
  --samples 2 ^
  --sub-lora .\hotpotqa_sub_pref_enhanced_300x900\sub ^
  --max-tokens 120 ^
  --sub-steps 3 ^
  --seed 123
```

结果：

| Sub | support_read_recall | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|
| Previous Preference / Stage2 Sub | 0.378 | 0.285 | 0.378 | 0.375 | 0.356 | 0.430 |
| Enhanced Preference Sub | 0.502 | 0.440 | 0.502 | 0.508 | 0.516 | 0.566 |

结论：

```text
Enhanced preference training 成功修复了 Sub retrieval：
support_read_recall 0.378 -> 0.502
reward              0.375 -> 0.508
```

这说明 MAS 在 enhanced 上的瓶颈定位正确：旧 Sub 不是不会工具调用，而是在更多 distractors 下 retrieval 不够强。

## Full MAS With Enhanced Sub

使用：

```text
Main:
  .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main

Sub:
  .\hotpotqa_sub_pref_enhanced_300x900\sub
```

评估 enhanced val 前 100 条、每题 2 samples：

```bash
python analyze_hotpotqa_mas_results.py ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --offset 0 ^
  --tasks 100 ^
  --samples 2 ^
  --main-lora .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main ^
  --sub-lora .\hotpotqa_sub_pref_enhanced_300x900\sub ^
  --max-tokens 120 ^
  --sub-steps 3 ^
  --seed 123
```

结果：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Old MAS Stage 2, enhanced multi-offset | 0.313 | 0.372 | 0.394 | 0.400 | 0.462 |
| MAS Stage 2 Main + Enhanced Sub, 100 tasks | 0.407 | 0.512 | 0.488 | 0.500 | 0.560 |

同一 enhanced 100 题上，对比 direct GRPO：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Direct Main GRPO 150x3 | 0.365 | 0.515 | 0.458 | 0.401 | 0.485 |
| MAS Stage 2 Main + Enhanced Sub | 0.407 | 0.512 | 0.488 | 0.500 | 0.560 |

关键结论：

```text
在 enhanced 100-task eval 上，MAS 首次超过 direct Main GRPO：
answer_f1 0.365 -> 0.407
reward    0.458 -> 0.488
best_answer_f1 0.401 -> 0.500
best_reward    0.485 -> 0.560
```

这是当前最重要的阶段性进展。它说明：

```text
1. 增强环境让 multi-agent 分工更有价值。
2. Sub retrieval 经过 enhanced preference training 后成为有效组件。
3. 冻结 Stage 2 Main + 强化后的 Sub，可以超过 direct Main GRPO。
```

当前最强 enhanced MAS checkpoint：

```text
Main:
  .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main

Sub:
  .\hotpotqa_sub_pref_enhanced_300x900\sub
```

下一步建议：

```text
1. 对 MAS Stage2+EnhancedSub 做完整 offsets 0,20,40,60,80 多切片评估。
2. 如果仍然超过 direct GRPO，再冻结 EnhancedSub 训练 Main。
3. 最后谨慎做 enhanced joint GRPO。
```

## Multi-Offset Eval: Stage2 Main + Enhanced Sub

对当前最强 enhanced MAS 组合做正式多 offset 评估：

```bash
python run_hotpotqa_eval_suite.py ^
  --suite mas ^
  --model-names stage2_main_enhanced_sub ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --offsets 0 20 40 60 80 ^
  --tasks 20 ^
  --samples 2 ^
  --out-dir .\hotpotqa_mas_eval_enhanced_sub_offsets ^
  --max-tokens 120 ^
  --sub-steps 3 ^
  --seed 123
```

逐切片结果：

| offset | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.456 | 0.537 | 0.527 | 0.574 | 0.617 |
| 20 | 0.399 | 0.450 | 0.469 | 0.483 | 0.538 |
| 40 | 0.447 | 0.512 | 0.515 | 0.520 | 0.569 |
| 60 | 0.392 | 0.537 | 0.482 | 0.528 | 0.585 |
| 80 | 0.296 | 0.487 | 0.405 | 0.377 | 0.469 |

Task-weighted average：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Direct Main GRPO 150x3 | 0.372 | 0.515 | 0.463 | 0.410 | 0.491 |
| MAS Stage2 Main + Enhanced Sub | 0.398 | 0.505 | 0.480 | 0.496 | 0.556 |

正式结论：

```text
MAS Stage2 Main + Enhanced Sub 在 enhanced 多 offset 评估上稳定超过 direct Main GRPO。
```

提升：

```text
answer_f1      +0.026
reward         +0.017
best_answer_f1 +0.086
best_reward    +0.065
```

这修正了前面的阶段判断：

```text
旧 MAS checkpoint 没有超过 direct；
但在 enhanced train 上重新训练 Sub retrieval 后，MAS 开始超过 direct。
```

关键原因：

```text
增强环境让 direct Main 的单体 search/read 压力变大；
Enhanced Sub preference 显著提升 retrieval；
Stage2 Main 已经具备利用 Sub evidence 的能力；
两者组合后形成了真正的 MAS 优势。
```

当前 best checkpoint：

```text
Main:
  .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main

Sub:
  .\hotpotqa_sub_pref_enhanced_300x900\sub
```

下一步：

```text
1. 冻结 Enhanced Sub，在 enhanced train 上继续训练 Main。
2. 只在 Main-only 继续提升稳定后，再尝试 enhanced joint。
3. 不建议直接 joint，因为之前 joint 容易损伤 Sub retrieval。
```

## Enhanced Main-only Follow-up

尝试冻结 Enhanced Sub，继续训练 Main。

第一版：

```bash
python grpo_hotpotqa_mas.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --tasks 50 ^
  --val-tasks 40 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --main-lora .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main ^
  --sub-lora .\hotpotqa_sub_pref_enhanced_300x900\sub ^
  --save-dir .\hotpotqa_mas_enhanced_mainonly_50x1 ^
  --lr 3e-6 ^
  --reward-threshold 0.35 ^
  --train-main ^
  --no-train-sub
```

结果：

```text
init val reward = 0.467
iter val reward = 0.421
```

该配置过拟合训练切片，validation 下降。

随后改为更保守配置：

```bash
python grpo_hotpotqa_mas.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --tasks 50 ^
  --val-tasks 40 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --main-lora .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main ^
  --sub-lora .\hotpotqa_sub_pref_enhanced_300x900\sub ^
  --save-dir .\hotpotqa_mas_enhanced_mainonly_conservative_50x1 ^
  --lr 1e-6 ^
  --reward-threshold 0.5 ^
  --train-main ^
  --no-train-sub
```

内部 validation：

```text
init val reward = 0.415
iter val reward = 0.433
saved best
```

外部 enhanced 100-task eval：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Stage2 Main + Enhanced Sub | 0.407 | 0.512 | 0.488 | 0.500 | 0.560 |
| Conservative Main-only 50x1 | 0.409 | 0.512 | 0.489 | 0.500 | 0.560 |

结论：

```text
保守 Main-only 没有破坏当前 best，但提升极小：
answer_f1 +0.002
reward    +0.001
```

这说明当前 Main-only 路线已经接近饱和。当前 best 可以更新为 conservative Main-only，但实际差异可以视作打平。

当前推荐 checkpoint：

```text
Main:
  .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\main

Sub:
  .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\sub
```

下一步建议：

```text
1. 不继续扩大 Main-only。
2. 如果继续训练，应优先提升 Sub retrieval 或做 carefully constrained joint。
3. 更可靠的下一步是提高 docs_per_task 到 50，验证 MAS 优势是否进一步扩大。
```

## Conservative Enhanced Joint Attempt

为了验证 joint 是否能在 staged best 基础上继续提升，尝试从当前 best 初始化做保守 joint：

```bash
python grpo_hotpotqa_mas.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --tasks 50 ^
  --val-tasks 40 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --main-lora .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\main ^
  --sub-lora .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\sub ^
  --save-dir .\hotpotqa_mas_enhanced_joint_conservative_50x1 ^
  --lr 1e-6 ^
  --reward-threshold 0.55 ^
  --train-main ^
  --train-sub
```

内部 validation：

```text
init val reward = 0.481
iter val reward = 0.404
updates main = 22
updates sub  = 22
```

因为 validation 下降，`best/` 没有更新，仍是训练前 staged best。

外部 enhanced 100-task eval 对 `main_step_1/sub_step_1`：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Staged best | 0.409 | 0.512 | 0.489 | 0.500 | 0.560 |
| Joint step1 | 0.359 | 0.500 | 0.451 | 0.440 | 0.514 |

Sub oracle 对 `sub_step_1`：

| Sub | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Enhanced Preference Sub | 0.502 | 0.440 | 0.502 | 0.508 | 0.525 | 0.516 | 0.566 |
| Joint step1 Sub | 0.480 | 0.384 | 0.482 | 0.465 | 0.515 | 0.441 | 0.510 |

结论：

```text
当前 conservative enhanced joint 失败。
它同时降低完整 MAS reward 和 Sub oracle reward。
```

这再次说明，当前阶段的最优策略不是 joint，而是 staged：

```text
Sub preference retrieval training
-> freeze Sub
-> Main-only conservative tuning
```

当前 best 仍然是：

```text
Main:
  .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\main

Sub:
  .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\sub
```

如果后续还要尝试 joint，需要改变 joint 机制，而不是简单继续降低 lr：

```text
1. 冻结 Sub action layers / 只训 Sub summary。
2. 对 Sub action 混入 preference replay，防止 retrieval 遗忘。
3. 对 Main/Sub 使用不同 reward threshold。
4. 或者使用真正 group advantage / DPO-style objective，而不是 best-of-group weighted SFT。
```

## Joint With Enhanced Sub Reward

为了解决 joint 中 Sub reward 过于接近 Main reward 的问题，`grpo_hotpotqa_mas.py` 新增：

```text
--sub-reward-mode enhanced
```

Enhanced Sub reward：

```text
sub_train_reward =
  0.40 * support_read_recall
+ 0.25 * sub_summary_answer_f1
+ 0.15 * sub_evidence_recall
+ 0.10 * read_precision
+ 0.05 * action_valid
+ 0.05 * no_duplicate_read
```

同时，在 Main/Sub joint 且 `sub_reward_mode=enhanced` 时，candidate selection 改为混合目标：

```text
0.55 * main_reward + 0.45 * sub_train_reward
```

训练命令：

```bash
python grpo_hotpotqa_mas.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --tasks 50 ^
  --val-tasks 40 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --main-lora .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\main ^
  --sub-lora .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\sub ^
  --save-dir .\hotpotqa_mas_enhanced_joint_subreward_50x1 ^
  --lr 1e-6 ^
  --reward-threshold 0.45 ^
  --best-metric reward ^
  --sub-reward-mode enhanced ^
  --train-main ^
  --train-sub
```

内部 validation：

```text
init val:
  reward        = 0.391
  answer_f1     = 0.301
  evidence      = 0.400
  sub_train     = 0.437
  sub_retrieval = 0.400
  sub_precision = 0.425

iter val:
  reward        = 0.478
  answer_f1     = 0.418
  evidence      = 0.425
  sub_train     = 0.481
  sub_retrieval = 0.425
  sub_precision = 0.438
  saved best
```

外部 enhanced 100-task eval：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Staged best | 0.409 | 0.512 | 0.489 | 0.500 | 0.560 |
| Joint enhanced-reward best | 0.361 | 0.490 | 0.451 | 0.446 | 0.517 |

Sub oracle：

| Sub | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Enhanced Preference Sub | 0.502 | 0.440 | 0.502 | 0.508 | 0.525 | 0.516 | 0.566 |
| Joint enhanced-reward Sub | 0.495 | 0.373 | 0.495 | 0.460 | 0.535 | 0.436 | 0.511 |

结论：

```text
Enhanced Sub reward 比旧 joint 更好地保护了 retrieval：
support_read_recall 0.502 -> 0.495
```

但它仍然损伤了：

```text
Sub summary answer_f1: 0.440 -> 0.373
Full MAS reward:       0.489 -> 0.451
```

所以当前 joint 的主要问题已经从“retrieval 被严重破坏”变成：

```text
Sub summary / Main-Sub output distribution 被 joint 更新扰动。
```

最终判断：

```text
仅靠更合理的 scalar Sub reward 还不够。
当前 best-of-group weighted SFT 形式的 joint 仍低于 staged training。
```

后续如果继续改 joint，建议加入 replay/约束：

```text
1. Sub action preference replay，保护 retrieval。
2. Sub summary SFT replay，保护 summary format 和 answer clue。
3. Main answer replay，保护 Main 使用 Sub result 的能力。
4. Main/Sub 分别使用不同 update threshold。
5. 或实现真正 GRPO advantage，而不是只做 best sample 加权 SFT。
```
