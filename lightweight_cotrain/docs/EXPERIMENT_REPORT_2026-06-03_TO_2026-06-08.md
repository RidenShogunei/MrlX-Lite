# 2026-06-03 至 2026-06-08 实验详细汇报

## 1. 范围与口径

本报告只统计北京时间 **2026 年 6 月 3 日（上周三）至 6 月 8 日** 完成的工作。

6 月 2 日已经完成的 Enhanced HotpotQA 环境和初版 dynamic MAS，只作为本期实验的前置基线，不计入本期成果。报告按照以下三条实验线展开：

1. HotpotQA 动态 MAS 的收尾、诊断和模型扩展；
2. 从 HotpotQA 迁移到 Plancraft，建立可执行 SFT/RL 基线；
3. 对 GRPO 实现与评测协议进行审计，并运行严格 policy-ratio GRPO。

本期共完成 28 个实验与工程提交，不含本报告提交。

---

## 2. 一页结论

### 2.1 本期真正完成了什么

- 动态 HotpotQA Main 已能稳定生成约 `1.8` 个 Sub 任务；
- dynamic synthesis SFT 将动态 MAS 平均 reward 从 `0.458` 提高到 `0.472`；
- 动态 MAS evidence 高于固定 MAS，但 answer synthesis 仍较弱；
- 完成 Qwen3.5-9B 的 SFT 和 dynamic MAS 小样本验证；
- 删除原 MrlX 项目代码，仓库收敛到当前自研实验；
- 接入 Plancraft 官方环境和 Oracle；
- 建立 structured `1 Main + 1 Sub` SFT200 基线，easy100 success 为 `0.40`；
- 证明早期所谓 GRPO 实际是 weighted SFT；
- 对齐 rollout、validation 和外部评测协议；
- 实现 old-policy ratio、clipping、reference KL 和多 policy epoch 的严格 GRPO；
- strict joint GRPO 首轮验证从 `0.55` 降到 `0.20`，更新被 validation 拒绝。

### 2.2 当前最重要的结论

1. **动态 MAS 的 SFT 原型成立，但动态 RL 尚未开始。**

   HotpotQA 中已经实现多个 Sub，但模型几乎不会选择 direct；Plancraft 中目前始终是固定 `1 Main + 1 Sub`。

2. **当前最可靠的 Plancraft 模型仍是 SFT200，不是 RL checkpoint。**

   SFT200 easy100 success 为 `0.40`。所有 weighted-SFT 和 strict GRPO 实验都没有稳定超过它。

3. **早期 RL 正收益被评测对齐和大样本实验推翻。**

   easy20 上曾出现 `0.50 -> 0.55`，但 easy100 变为 `0.40 -> 0.38`；统一评测后，低强度更新从 `0.55` 降到 `0.25`。

4. **严格 GRPO 已经“能训练”，但还没有“训练对”。**

   optimizer、ratio、KL、clipping 都实际运行，但 joint validation 从 `0.55` 降到 `0.20`。当前瓶颈转向 step-level credit、reward alignment 和 Main/Sub 非平稳耦合。

---

## 3. 6 月 3 日：动态 HotpotQA、项目清理与 9B 尝试

### 3.1 Dynamic mixture replay

本日首先修正 dynamic MAS 的输入和训练数据。

此前 Main 被要求生成类似：

```text
Find evidence from document D16
```

但 plan prompt 中没有提供文档目录，Main 实际只能猜测文档 ID。修正后，Main 在规划前能看到：

```text
Available documents:
D01: title
D02: title
...
```

同时生成 mixture replay 数据：

```text
300 enhanced HotpotQA tasks
4200 SFT samples
1200 Main samples
3000 Sub samples
fixed protocol replay + dynamic focused subtasks
```

训练配置：

```text
epochs = 1
lr = 5e-5
max_length = 1536
Main loss = 0.0298
Sub loss = 0.0721
```

20-task dynamic validation：

| 模型 | avg subtasks | answer F1 | evidence | reward |
|---|---:|---:|---:|---:|
| Dynamic mixture v3 | 1.800 | 0.224 | 0.675 | 0.392 |
| Fixed staged baseline | - | 0.456 | 0.537 | 0.527 |

**当日判断：**

- dynamic planning 和 evidence selection 已开始工作；
- evidence `0.675` 已高于固定 MAS；
- answer F1 只有 `0.224`，主要瓶颈从 retrieval 转向 final synthesis。

### 3.2 Dynamic synthesis Main-only SFT

为避免继续扰动 Sub，冻结 dynamic mixture Sub，仅训练 Main 综合多个 focused Sub results。

数据与训练：

```text
500 Main-only synthesis samples
start Main = dynamic mixture v3
Sub = frozen
epochs = 1
lr = 3e-5
loss = 0.2391
```

20-task 结果：

| 模型 | avg subtasks | answer F1 | evidence | reward |
|---|---:|---:|---:|---:|
| Mixture v3 | 1.800 | 0.224 | 0.675 | 0.392 |
| Synthesis 300 | 1.850 | 0.254 | 0.662 | 0.410 |
| Synthesis 500 | 1.825 | 0.316 | 0.700 | 0.461 |

这说明 targeted Main-only SFT 是有效的：它没有牺牲 evidence，并明显提高了最终回答。

### 3.3 仓库清理

完成两次项目清理：

- 整理当前 HotpotQA MAS 代码结构；
- 删除原作者的 `MrlX-DeepResearch`、`MrlX-SelfRewarding`、`MrlX-TakesTwo` 等目录；
- 保留当前实际训练和评测使用的 lightweight cotrain 代码。

这一步解决了仓库中“原项目代码”和“当前实验代码”混杂的问题。

### 3.4 Qwen3.5-9B 实验

将 HotpotQA 单体和动态 MAS 迁移到 Qwen3.5-9B，完成 SFT 与小样本验证。

10 tasks、每题 2 samples：

| 模型 | tool valid | answer F1 | evidence | reward | best reward |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B base | 0.900 | 0.200 | 0.700 | 0.370 | 0.511 |
| Qwen3.5-9B SFT | 1.000 | 0.542 | 0.650 | 0.609 | 0.660 |
| Qwen3.5-9B dynamic MAS | 0.850 | 0.458 | 0.725 | 0.551 | 0.657 |

Dynamic MAS 额外指标：

```text
direct_rate = 0.150
avg_subtasks = 1.000
```

**结论边界：**

- 9B SFT 小样本结果很强；
- dynamic MAS evidence 较高；
- 只有 10 个任务，且未做多 offset、等训练数据和等推理预算对比；
- 因此不能据此得出“9B 动态 MAS 已优于现有系统”。

### 3.5 TRL GRPO 尝试

新增基于 Hugging Face TRL `GRPOTrainer` 的 HotpotQA 脚本，并修正 reward 函数，使其能读取：

```text
gold_answer
gold_doc_ids
```

但 TRL 标准 trainer 主要面向单次 prompt-response。当前系统包含：

- Main plan；
- 多轮 Sub search/read；
- Main final answer；
- Main/Sub 两个 adapter；
- 不同角色的奖励。

因此直接套用标准 `GRPOTrainer` 难以表达完整 MAS 轨迹和分角色信用，本路线没有继续作为主训练线。

### 3.6 6 月 3 日阶段判断

当日形成了两个方向：

1. 动态 MAS 已经不是“不会分 Sub”，而是“会分，但不会稳定综合”；
2. 单纯换大模型可能提高上限，但不能解决训练算法和信用分配问题。

---

## 4. 6 月 4 日：动态 MAS 诊断、GRPO 审计与 Plancraft 迁移

### 4.1 Dynamic multi-offset validation

对 mixture v3 和 synthesis 500 做 5 个 offset：

```text
offsets = 0, 20, 40, 60, 80
10 tasks per offset
2 samples per task
```

平均结果：

| 模型 | avg subtasks | answer F1 | evidence | reward | best reward |
|---|---:|---:|---:|---:|---:|
| Dynamic mixture v3 | 1.840 | 0.318 | 0.675 | 0.458 | 0.516 |
| Dynamic synthesis 500 | 1.830 | 0.344 | 0.655 | 0.472 | 0.516 |
| Fixed staged baseline | - | 0.413 | 0.495 | 0.488 | 0.575 |

**结果解释：**

- synthesis 在平均值上确实改善 dynamic MAS；
- dynamic evidence `0.655` 高于 fixed 的 `0.495`；
- dynamic answer F1 `0.344` 低于 fixed 的 `0.413`；
- dynamic reward `0.472` 仍低于 fixed 的 `0.488`。

这进一步确认：动态线的主要问题不是找不到证据，而是不能稳定把多个局部结果转成最终答案。

### 4.2 Offset 40 failure trace

新增细粒度失败分析：

```text
plan_support_recall
read_support_recall
sub_summary_evidence_recall
sub_summary_answer_f1
final_answer_f1
final_evidence
duplicate_reads
```

Synthesis 500 在 offset 40 的一次 trace：

| 指标 | 值 |
|---|---:|
| plan support recall | 0.550 |
| read support recall | 0.700 |
| Sub summary evidence recall | 0.650 |
| Sub summary answer F1 | 0.388 |
| final answer F1 | 0.475 |
| final evidence | 0.700 |
| duplicate reads | 0.000 |

多个失败样本已经读取两篇 gold documents，但比较、多跳关系或最终实体仍答错。

### 4.3 Sub evidence replay 实验

第一版仅训练 Sub evidence summary：

```text
500 tasks
1000 Sub samples
lr = 3e-5
```

Offset 40：

```text
answer F1 = 0.046
evidence = 0.625
reward = 0.257
```

这是明显的灾难性遗忘：Sub 更像摘要器，但丢失了 rollout 所需的 action/read/answer 行为。

第二版混入 action replay：

```text
4500 Sub samples
fixed action replay
focused action replay
evidence summaries
lr = 2e-5
```

Offset 40：

| Sub | answer F1 | evidence | reward |
|---|---:|---:|---:|
| Original mixture Sub | 0.137 | 0.675 | 0.331 |
| Evidence replay Sub | 0.205 | 0.650 | 0.374 |

但多 offset 平均：

```text
answer F1 = 0.088
evidence = 0.690
reward = 0.300
```

**最终结论：** replay 修复了特定切片，却破坏了整体答案能力，不能替换原 Sub。

### 4.4 Main verifier SFT

冻结原 dynamic Sub，训练 Main 面对：

- gold evidence-only results；
- wrong-first conflicting result；
- distractor result；
- partial evidence。

训练：

```text
500 tasks x 3 variants
1500 Main samples
lr = 2e-5
loss = 0.0845
```

Offset 40：

| Main | answer F1 | reward | best answer F1 | best reward |
|---|---:|---:|---:|---:|
| Synthesis | 0.137 | 0.331 | 0.140 | 0.338 |
| Verifier | 0.312 | 0.454 | 0.500 | 0.600 |

多 offset 平均：

| Main | answer F1 | evidence | reward | best answer F1 | best reward |
|---|---:|---:|---:|---:|---:|
| Synthesis | 0.344 | 0.655 | 0.472 | 0.394 | 0.516 |
| Verifier | 0.330 | 0.595 | 0.450 | 0.455 | 0.550 |

Verifier 提高难例和 best-of 表现，但平均单样本退化。它更适合 reranking 或 best-output distillation，不适合直接替换 synthesis Main。

### 4.5 HotpotQA GRPO 算法审计

检查旧训练循环后发现：

```text
group sampling
-> select best candidate
-> teacher-force winner
```

它实际上是 winner-only reward-filtered SFT，而非严格 GRPO。

新增 advantage objective：

```text
advantage = normalize(reward within task group)
positive advantage -> increase trajectory probability
negative advantage -> decrease trajectory probability
```

Smoke：

```text
init val reward = 0.200
train reward = 0.300
Main updates = 2
Sub updates = 2
final val reward = 0.200
```

该版本修复了坏样本被完全丢弃的问题，但仍缺少：

- old policy ratio；
- ratio clipping；
- explicit reference KL。

所以它只是 advantage-based policy update，不是最终严格 GRPO。

### 4.6 迁移到 Plancraft

同日决定将主线迁移到 Plancraft，原因是：

- 环境确定且可执行；
- 动作是否合法可以直接验证；
- 有官方 Oracle planner；
- 有明确 terminal success；
- 可计算每一步是否推进目标；
- 比 HotpotQA 更容易定位 reward 和 optimizer 问题。

完成：

```text
plancraft_environment.py
analyze_plancraft_results.py
analyze_plancraft_mas_results.py
patch_plancraft_windows.py
```

Oracle sanity check 成功率达到 100%，说明环境封装和任务链路可用。

### 4.7 Plancraft SFT 与初版 MAS

生成 Oracle SFT：

```text
Sub:
  state + history -> local advice

Main:
  state + history + Sub advice -> executable action
```

当前架构从一开始就是：

```text
固定 1 Main + 固定 1 Sub
```

这不是 HotpotQA 的动态 `0 / 1 / N Sub` 架构。

当日也建立了初版 GRPO-style trainer，并开始拆分 Main/Sub reward。

---

## 5. 6 月 5 日：Plancraft SFT 基线、Reward 与扩大实验

### 5.1 Structured Sub interface

Sub 输出从单一 action 改为：

```text
<subgoal>...</subgoal>
<reason>...</reason>
<action>...</action>
```

SFT50 easy5：

| 版本 | success | avg steps | invalid rate |
|---|---:|---:|---:|
| Action-only | 0.40 | 6.8 | 0.28 |
| Structured | 0.20 | 8.4 | 0.04 |

结构化接口将非法动作率从 `0.28` 降到 `0.04`，但 50-task 数据不足，成功率下降。

### 5.2 Structured short-history SFT200

关键改动：

```text
tasks = 200
Main samples = 1430
Sub samples = 1430
history steps = 3
max length = 2048
```

缩短 history 后，全部目标 token 都能保留，不再因上下文过长截断关键 action。

结果：

| Eval | success | efficiency | avg steps | invalid |
|---|---:|---:|---:|---:|
| easy5 | 0.40 | - | 4.60 | 0.070 |
| easy100 | 0.40 | 0.182 | 5.57 | 0.097 |

这是本期建立的最可靠 checkpoint：

```text
plancraft_mas_structured_short_sft_200x1/main_agent
plancraft_mas_structured_short_sft_200x1/sub_agent
```

### 5.3 Main/Sub reward 拆分

Main reward：

```text
terminal success
valid action
oracle action match
oracle state progress
step efficiency
```

Sub reward：

```text
partial terminal reward
structured action validity
oracle action match
oracle state progress
Main/Sub agreement
step efficiency
```

State progress 使用 Oracle 剩余距离：

```text
progress = (distance_before - distance_after) / distance_before
```

这比 exact action match 更宽松，可以奖励不同但确实推进状态的动作。

### 5.4 Per-step GRPO-style

早期 objective：

```text
loss = advantage * cross_entropy(rollout response)
```

它有 group-relative advantage，但没有 policy ratio 和 KL，本质仍是 advantage-weighted SFT。

20 train / 20 val / group 4，每个 adapter 约 400 次更新：

| 模型 | easy20 success | avg steps | invalid |
|---|---:|---:|---:|
| SFT200 | 0.50 | 5.05 | 0.077 |
| Per-step weighted update | 0.45 | 5.25 | 0.083 |

更新过密，验证性能下降。

### 5.5 Group-batched update + SFT replay

随后将更新改为：

```text
accumulate one group
one Main optimizer step
one Sub optimizer step
small SFT replay
```

easy20：

| 模型 | success | avg steps | invalid |
|---|---:|---:|---:|
| SFT200 | 0.50 | 5.05 | 0.077 |
| Batch + replay | 0.55 | 4.80 | 0.034 |

这是本期第一次看起来有正收益的“RL”结果，但当天扩大到 easy100 后：

| 模型 | success | efficiency | avg steps | invalid |
|---|---:|---:|---:|---:|
| SFT200 | 0.40 | 0.182 | 5.57 | 0.097 |
| Batch + replay | 0.38 | 0.169 | 5.68 | 0.078 |

配对统计：

```text
updated-only solves = 7
SFT-only solves = 9
difference = -0.02
bootstrap 95% CI = [-0.10, 0.06]
McNemar p = 0.804
```

**修正结论：**

- easy20 的 `+0.05` 是小样本波动；
- 更新降低了非法动作率；
- 但没有提高总体任务成功率。

---

## 6. 6 月 6 日：独立训练集与评测协议对齐

### 6.1 Independent 50-task rollout

为了排除 RL tasks 与 SFT200 数据重合：

```text
train offset = 200
RL tasks = 50
group size = 4
rollouts = 200
```

easy100：

| 模型 | success | efficiency | avg steps | invalid |
|---|---:|---:|---:|---:|
| SFT200 | 0.40 | 0.182 | 5.57 | 0.097 |
| Independent 50-task update | 0.38 | 0.162 | 5.85 | 0.092 |

扩大到 200 条独立 rollout 后依然没有超过 SFT。

这否定了“之前只是 RL 数据量不够或训练数据重叠”的简单解释。

### 6.2 发现评测协议不一致

Trainer 内部 validation：

```text
temperature = 0.8
best-of-N
max_steps = 8
```

外部 evaluator：

```text
temperature = 0.2
single sample
max_steps = 10
fixed seed
```

这会导致 trainer 选择一个“高温多采样较好、低温单样本较差”的 checkpoint。

### 6.3 Aligned validation

统一为：

```text
eval temperature = 0.2
top_p = 0.9
repetition penalty = 1.05
max_steps = 10
samples = 1
seed = 123
best metric = success rate
```

SFT200 aligned easy20：

```text
success = 0.55
```

评测对齐使后续所有 checkpoint 都必须在同一协议下与这个起点比较。

---

## 7. 6 月 8 日：低强度消融与严格 GRPO

6 月 7 日没有新的实验提交。6 月 8 日继续完成训练强度消融和严格 GRPO。

### 7.1 Low-strength Main-only 与 joint

为了验证是不是学习率、更新次数或 Sub 联合更新导致退化，运行低强度版本。

Aligned easy20：

| 模型 | before | after |
|---|---:|---:|
| Low-strength Main-only | 0.55 | 0.25 |
| Low-strength joint | 0.55 | 0.25 |

**结论：**

- 简单降低更新强度不能解决问题；
- Main-only 已经出现同等退化；
- 因此退化不主要由 Sub 联合更新引起；
- 更可能是 Main 的 weighted teacher-forcing objective 与 validation success 不一致。

注意：这仍是旧 weighted-SFT objective 下的消融，不是 strict Main-only GRPO。

### 7.2 严格 policy-ratio GRPO 实现

新增：

```text
old policy token logprob
frozen Main reference
frozen Sub reference
current/old probability ratio
clipped surrogate objective
reference-policy KL
multiple policy epochs
group gradient accumulation
aligned validation
```

核心目标：

```text
ratio = exp(logpi_current - logpi_old)

policy_loss = -min(
  ratio * advantage,
  clip(ratio, 1-epsilon, 1+epsilon) * advantage
)

loss = policy_loss + beta * KL(current || reference)
```

数值测试确认：

- 正 advantage 提高对应 token 概率；
- 负 advantage 降低对应 token 概率；
- ratio 超界时发生 clipping；
- current/reference 初始 log-prob 相同；
- 第二个 policy epoch 后 ratio 和 KL 开始变化。

### 7.3 Strict smoke

2-task smoke：

```text
policy epochs = 2
Main updates = 4
Sub updates = 4
Main ratio ~= 1.0000
Sub ratio ~= 0.9996
Sub KL = 0.000382
```

这证明 strict 训练链路不是空跑。

### 7.4 Strict joint 20-task

配置：

```text
train offset = 200
tasks = 20
group size = 4
policy epochs = 2
clip epsilon = 0.2
KL beta = 0.01
lr = 1e-7
aligned validation = true
```

训练与验证：

| 指标 | 结果 |
|---|---:|
| val before | 0.55 |
| train success | 0.25 |
| Main optimizer steps | 40 |
| Sub optimizer steps | 40 |
| Main policy loss | 0.2346 |
| Sub policy loss | 0.2073 |
| Main KL | 约 0 |
| Sub KL | 0.000233 |
| Main clip fraction | 0.0000 |
| Sub clip fraction | 0.0016 |
| val after | 0.20 |

Validation 拒绝更新后的 checkpoint，best 仍保留 SFT200。

### 7.5 Strict 结果的正确解释

这次失败不能归因于“GRPO 没有运行”：

- optimizer step 数正常；
- policy loss 非零；
- Sub ratio、KL 和 clip fraction 有变化；
- validation 明确检测到性能下降。

更可能的原因：

1. episode reward 被整条轨迹共享，token/step credit 太粗；
2. terminal success 稀疏，局部 shaping 主导 advantage；
3. Oracle progress 奖励局部接近，但不保证最终计划仍可完成；
4. Main/Sub agreement 可能鼓励复制，而非互补；
5. 同时更新 Main/Sub 导致接口分布变化；
6. group 内有效动作差异可能不足，advantage 信号噪声较高；
7. 20-task rollout 对 held-out 任务的覆盖仍不足。

### 7.6 未完成实验

Strict Main-only 20-task 实验运行时间较长，中途被人工停止。残留进程已清理，partial 输出不能用于结论。

因此当前还缺少严格条件下的：

- Main-only；
- Sub-only；
- no-update rollout control；
- easy100 strict checkpoint 评测。

由于 strict joint 在 aligned easy20 已经退化，没有必要继续把该 checkpoint 扩评到 easy100。

---

## 8. 本期完整结果汇总

### 8.1 HotpotQA dynamic MAS

| 实验 | Eval | answer F1 | evidence | reward | 结论 |
|---|---|---:|---:|---:|---|
| Mixture v3 | hard20 | 0.224 | 0.675 | 0.392 | Evidence 提升，综合较弱 |
| Synthesis 500 | hard20 | 0.316 | 0.700 | 0.461 | Main-only synthesis 有效 |
| Mixture v3 | multi-offset 50 | 0.318 | 0.675 | 0.458 | 动态规划稳定生成约 1.8 Sub |
| Synthesis 500 | multi-offset 50 | 0.344 | 0.655 | 0.472 | 平均提升但仍低于 fixed |
| Fixed staged | multi-offset 50 | 0.413 | 0.495 | 0.488 | 当前 HotpotQA 更稳 |
| Evidence replay Sub | multi-offset 50 | 0.088 | 0.690 | 0.300 | 局部修复、整体遗忘 |
| Verifier Main | multi-offset 50 | 0.330 | 0.595 | 0.450 | best-of 提升、均值下降 |

### 8.2 Plancraft

| 实验 | Eval | success | invalid | 结论 |
|---|---|---:|---:|---|
| Action-only SFT50 | easy5 | 0.40 | 0.280 | 会解题但格式不稳 |
| Structured SFT50 | easy5 | 0.20 | 0.040 | 格式改善，数据不足 |
| Structured SFT200 | easy100 | **0.40** | 0.097 | 当前可靠基线 |
| Per-step weighted update | easy20 | 0.45 | 0.083 | 低于 SFT |
| Batch + replay | easy20 | 0.55 | 0.034 | 小样本表面提升 |
| Batch + replay | easy100 | 0.38 | 0.078 | 提升未复现 |
| Independent 50-task | easy100 | 0.38 | 0.092 | 扩量仍未提升 |
| Low-strength Main-only | aligned easy20 | 0.25 | - | 从 0.55 下降 |
| Low-strength joint | aligned easy20 | 0.25 | - | 从 0.55 下降 |
| Strict joint GRPO | aligned easy20 | 0.20 | 0.048 | 从 0.55 明显下降 |

---

## 9. 结论如何发生变化

### 6 月 3 日时

我们认为 dynamic MAS 的主要问题是 synthesis，可以通过 targeted SFT 修复；同时考虑使用更大模型和标准 TRL GRPO。

### 6 月 4 日时

我们确认：

- synthesis 确实可改善；
- 但 verifier、Sub replay 会产生明显能力权衡；
- HotpotQA reward 很难把 retrieval、summary、verification、routing 一次性覆盖；
- 旧 GRPO 算法不严格；
- 因此迁移到更确定的 Plancraft。

### 6 月 5 日时

Plancraft SFT200 建立，batch+replay 在 easy20 出现 `+0.05`，一度看起来 RL 可能有效。

### 6 月 6 日时

easy100 和独立 50-task 实验推翻了小样本乐观结论；同时发现 validation 协议不一致。

### 6 月 8 日时

统一评测和 strict GRPO 给出更明确的负结果：

- 旧更新是 weighted SFT；
- 小学习率不能修复；
- strict GRPO 确实更新，但 joint performance 更差；
- 当前问题是 credit/reward alignment，而非单纯数据量或代码没跑。

---

## 10. 当前系统到底是什么

### HotpotQA

已经存在动态 MAS 原型：

```text
Main 可以生成多个 subtasks
多个 Sub instance 共享同一个 Sub adapter
每个 Sub history 独立
Main 汇总 Sub results
```

但：

- direct rate 接近 0；
- 子任务数集中在约 1.8；
- 只完成 SFT 和 evaluation；
- 没有完成严格 dynamic GRPO。

### Plancraft

当前是固定 MAS：

```text
1 Main
1 Sub
Sub 每一步给建议
Main 每一步执行一个动作
```

没有动态 agent 数量，也没有并行多个 Sub。

因此当前不能说“新的 Plancraft 实验已经验证动态 MAS”。它验证的是固定双角色架构下的 SFT 和 GRPO。

---

## 11. 目前最可信的科学结论

### 已经成立

1. Dynamic routing 行为可以通过 SFT 学到。
2. 多 Sub 能提高 evidence recall，但不自动提高最终答案。
3. 单独强化某项 Sub 能力会造成其他能力遗忘。
4. Plancraft structured SFT200 是有效且可复现的基线。
5. 早期 GRPO-style 结果不能作为严格 RL 证据。
6. 扩大 weighted-SFT rollout 数量没有超过 SFT。
7. 统一评测后，低强度 weighted update 仍会退化。
8. Strict policy-ratio GRPO 已实现并实际更新参数。
9. Strict joint GRPO 当前没有提升任务成功率。

### 尚未成立

1. RL 能稳定超过 Plancraft SFT200；
2. 联合训练优于 staged 或 Main-only；
3. 当前 reward 能支持动态 `0 / 1 / N Sub`；
4. Qwen3.5-9B dynamic MAS 优于固定 MAS；
5. 增加更多 RL 数据即可解决 strict GRPO 退化；
6. 当前项目已经复现 M-GRPO 论文结论。

---

## 12. 下一步应当做什么

下一步不建议继续换 reward 权重后直接跑长实验。应先完成最小因果矩阵：

| 实验 | Main | Sub | 目的 |
|---|---|---|---|
| No-update control | frozen | frozen | 测量纯采样方差 |
| Strict Main-only | train | frozen | 判断 Main policy update 是否有益 |
| Strict Sub-only | frozen | train | 判断 Sub 局部策略是否可独立改善 |
| Strict joint | train | train | 测量联合非平稳性 |

所有实验使用：

```text
同一 SFT200 起点
同一 train offset
同一 rollout group
同一 aligned easy20
同一 seed 和 decoding
```

同时新增每步日志：

- Oracle distance before/after；
- valid / invalid / no-op；
- terminal success；
- Main reward components；
- Sub reward components；
- unique action rate；
- group reward standard deviation；
- zero-advantage group rate；
- positive/negative advantage 数量；
- ratio、KL、clip fraction。

若 Main-only 仍下降，应先改 step-level advantage，不扩大训练：

```text
step advantage =
  state progress delta
  + valid transition
  - invalid/no-op
  + terminal success return
```

若 Main-only 稳定而 joint 下降，再处理 Main/Sub 接口漂移与 Sub reward。

只有固定 `1 Main + 1 Sub` 的 strict GRPO 能在 aligned validation 不下降后，才值得把 Plancraft 扩展到动态 `0 / 1 / N Sub`。

---

## 13. 当前阶段判断

从 6 月 3 日到 6 月 8 日，项目完成了从“动态 MAS 行为调试”到“可执行 benchmark 和严格 RL 审计”的迁移。

最重要的成果不是得到了一个更高分的 RL checkpoint，而是排除了几个错误解释：

- 不是只要增加 Sub 数量就会更强；
- 不是只要增加 rollout 数量就会提升；
- 不是只要降低学习率就能稳定；
- 不是此前所有名为 GRPO 的实验都属于真正 GRPO；
- 不是当前失败源于训练循环完全没更新。

当前准确进度为：

> HotpotQA 动态 MAS 已完成 SFT 原型和能力分解；Plancraft 已完成固定双角色 SFT 基线、评测对齐和严格 GRPO 实现。严格联合 RL 尚未超过 SFT，下一阶段必须先完成 Main-only、Sub-only、joint 和 no-update 的同协议消融。
