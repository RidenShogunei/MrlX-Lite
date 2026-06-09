# Plancraft Multi-Agent RL 阶段实验报告

更新时间：2026-06-08

## 1. 摘要

本阶段将实验主线从 HotpotQA/数学任务迁移到了开源、可执行、带官方
oracle planner 的 Plancraft 环境，目标是验证：

```text
Structured Sub -> Main -> Plancraft Env
```

其中：

- Sub 输出局部目标、理由和建议动作；
- Main 根据环境状态和 Sub 建议输出最终可执行动作；
- 环境执行 Main 动作；
- Main/Sub 使用独立奖励进行联合训练。

目前最可靠的结论如下：

1. Plancraft 已完整接入，官方 oracle 在环境 sanity check 中成功率为 100%。
2. Structured SFT 是有效的。将训练数据扩大到 200 个任务并缩短 history 后，
   easy100 成功率达到 40%。
3. 早期所谓 GRPO 实际是 advantage-weighted cross entropy，并非严格 GRPO。
4. 早期 GRPO 能降低非法动作率，但在 easy100 上没有超过 SFT。
5. 已实现 token-level clipped policy-ratio GRPO、old policy logprob、冻结
   reference adapter 和 reference KL。
6. 严格 joint GRPO 的首轮 20-task 实验仍使 aligned validation 从 55% 降到
   20%，因此当前没有证据证明 RL 提升了总体任务求解能力。
7. 当前最可靠、应保留的 checkpoint 仍是 structured SFT200。

## 2. 环境与架构

### 2.1 Plancraft

Plancraft 提供：

- Minecraft crafting 状态与动作空间；
- 可执行环境；
- 官方任务划分；
- 官方 oracle planner/subplans；
- 成功、终止和最优路径信息。

当前环境封装：

```text
plancraft_environment.py
```

支持：

```text
reset()
step(action)
oracle_subplans()
result()
```

Windows 安装包路径解析问题通过以下脚本修复：

```text
patch_plancraft_windows.py
```

### 2.2 Multi-Agent 架构

模型结构：

```text
shared Qwen2.5-1.5B-Instruct base
  + Main LoRA
  + Sub LoRA
```

交互流程：

```text
Observation + history
        |
        v
Structured Sub
  <subgoal>...</subgoal>
  <reason>...</reason>
  <action>...</action>
        |
        v
Main
  executable move/smelt/impossible action
        |
        v
Plancraft environment
```

环境只执行 Main 动作。Sub 的作用是提供局部规划与状态解释。

## 3. 数据与 SFT

### 3.1 Oracle SFT 数据

数据生成器：

```text
generate_plancraft_mas_sft_data.py
```

官方 oracle 在每个状态给出后续动作。每个 oracle step 生成：

```text
Sub sample:
  current state + history
  -> structured subgoal/reason/action

Main sample:
  current state + history + structured Sub advice
  -> executable action
```

### 3.2 Action-only SFT50

数据：

```text
50 tasks
317 Main samples
317 Sub samples
```

easy5：

| Metric | Value |
|---|---:|
| Success | 0.400 |
| Avg steps | 6.800 |
| Invalid action rate | 0.280 |

该版本能够求解简单任务，但非法 slot、重复动作和状态跟踪问题明显。

### 3.3 Structured SFT50

Sub 改为：

```text
<subgoal>...</subgoal>
<reason>...</reason>
<action>...</action>
```

easy5：

| Metric | Value |
|---|---:|
| Success | 0.200 |
| Avg steps | 8.400 |
| Invalid action rate | 0.040 |

结构化接口显著改善动作格式，但 50-task 数据不足以让 Main 稳定使用新协议。

### 3.4 Structured Short-History SFT200

关键改动：

```text
--history-steps 3
--max-length 2048
```

数据规模：

```text
200 tasks
1430 Main samples
1430 Sub samples
```

训练样本保留率：

```text
Main: 1430 / 1430
Sub:  1430 / 1430
```

easy5：

| Metric | Value |
|---|---:|
| Success | 0.400 |
| Avg steps | 4.600 |
| Invalid action rate | 0.070 |

完整 easy100：

| Metric | Value |
|---|---:|
| Success | **0.400** |
| Solved | **40 / 100** |
| Efficiency | 0.182 |
| Avg steps | 5.570 |
| Invalid action rate | 0.097 |

当前最可靠 checkpoint：

```text
plancraft_mas_structured_short_sft_200x1/main_agent
plancraft_mas_structured_short_sft_200x1/sub_agent
```

## 4. 奖励设计

### 4.1 Main Reward

当前 Main reward：

```text
success
+ valid action
+ oracle action match
+ oracle state progress
- step penalty
```

### 4.2 Sub Reward

当前 Sub reward：

```text
partial global success
+ structured action validity
+ oracle action match
+ oracle state progress
+ Sub/Main agreement
- step penalty
```

### 4.3 State Progress

使用官方 planner 计算状态距离：

```text
before = 当前状态 oracle 剩余步数
执行 Main 动作
after = 下一状态 oracle 剩余步数

progress = (before - after) / before
```

相较 exact action match，该指标能够奖励真正缩短剩余计划的动作。

## 5. 早期 GRPO-Style 实验

### 5.1 必须澄清的算法问题

早期更新为：

```text
loss = advantage * cross_entropy(rollout response)
```

它具有：

- group rollout；
- group-relative advantage；
- Main/Sub 独立奖励。

但缺少：

- old policy logprob；
- current/old probability ratio；
- PPO/GRPO clipping；
- reference policy KL。

因此本报告将其称为：

```text
GRPO-style weighted-SFT
```

而不是严格 GRPO。

### 5.2 Per-Step Update

最初每个 trajectory step 都执行一次 optimizer step。

20 train / 20 val / group 4 实验中，每个 adapter 产生约 400 次更新。

统一 easy20：

| Model | Success | Avg steps | Invalid |
|---|---:|---:|---:|
| SFT200 | 0.500 | 5.050 | 0.077 |
| Per-step GRPO-style | 0.450 | 5.250 | 0.083 |

结论：更新过密，验证性能下降。

### 5.3 Group-Batched Update + SFT Replay

改进：

```text
每个 group 累积梯度
Main/Sub 每组各 optimizer.step 一次
混入少量 SFT replay
```

easy20：

| Model | Success | Avg steps | Invalid |
|---|---:|---:|---:|
| SFT200 | 0.500 | 5.050 | 0.077 |
| Batch + replay | 0.550 | 4.800 | 0.034 |

该结果一度表现为正收益，但完整 easy100 未复现。

easy100：

| Model | Success | Efficiency | Avg steps | Invalid |
|---|---:|---:|---:|---:|
| SFT200 | **0.400** | **0.182** | **5.570** | 0.097 |
| Batch + replay | 0.380 | 0.169 | 5.680 | **0.078** |

配对统计：

```text
GRPO-only solves = 7
SFT-only solves = 9
difference = -0.020
bootstrap 95% CI = [-0.100, +0.060]
McNemar p = 0.804
```

结论：

- easy20 的 +0.05 是小样本波动；
- GRPO-style 更新降低非法动作率；
- 但没有提高总体成功率。

### 5.4 50-Task Independent RL

为了避免与 SFT200 数据重合：

```text
train offset = 200
RL tasks = 50
group size = 4
rollouts = 200
```

easy100：

| Model | Success | Efficiency | Avg steps | Invalid |
|---|---:|---:|---:|---:|
| SFT200 | **0.400** | **0.182** | **5.570** | 0.097 |
| 50-task GRPO-style | 0.380 | 0.162 | 5.850 | **0.092** |

扩大 RL 数据到 50 个独立任务后，仍未超过 SFT。

## 6. 评测协议修正

早期 trainer validation 使用：

```text
temperature = 0.8
best-of-N
max_steps = 8
```

外部 evaluator 使用：

```text
temperature = 0.2
single sample
max_steps = 10
fixed seed
```

两者不一致，导致内部 validation 可能选择外部表现更差的 checkpoint。

当前 aligned validation：

```text
rollout temperature = 0.8
eval temperature = 0.2
eval top_p = 0.9
eval repetition penalty = 1.05
eval max_steps = 10
eval samples = 1
eval seed = 123
best metric = success_rate
```

SFT200 aligned easy20：

```text
success = 0.550
```

低强度 GRPO-style Main-only 和 joint 都从 0.550 降至 0.250，说明主要损伤来自
Main 更新，而不是 Sub 联合更新。

## 7. 严格 Policy-Ratio GRPO

### 7.1 已实现组件

当前未提交工作区版本已实现：

```text
old policy token logprob
frozen Main reference adapter
frozen Sub reference adapter
current/old token probability ratio
clipped surrogate objective
reference-policy KL
multiple policy epochs
group-level gradient accumulation
SFT replay
aligned validation
```

目标函数：

```text
ratio = exp(logpi_current - logpi_old)

policy_loss = -min(
  ratio * advantage,
  clip(ratio, 1-epsilon, 1+epsilon) * advantage
)

loss = policy_loss + beta * KL(current || reference)
```

### 7.2 数值测试

已验证：

- 正 advantage 的梯度提高 rollout token 概率；
- 负 advantage 的梯度降低 rollout token 概率；
- ratio 超过 clipping 区间后 surrogate 被截断；
- current/reference 初始 token logprob 完全一致；
- 初始 reference KL 为 0；
- 第二个 policy epoch 后 ratio 会偏离 1，KL 开始非零。

### 7.3 Strict GRPO Smoke

2-task smoke：

```text
policy epochs = 2
Main/Sub updates = 4 / 4
Main ratio ~= 1.0000
Sub ratio ~= 0.9996
Sub KL = 0.000382
```

说明 strict policy-ratio、reference KL 和多 epoch 更新链路已经实际运行。

### 7.4 Strict Joint GRPO 20-Task

配置：

```text
train offset = 200
tasks = 20
group size = 4
policy epochs = 2
policy clip = 0.2
KL beta = 0.01
lr = 1e-7
aligned validation = true
```

结果：

```text
val before = 0.550
train success = 0.250
Main/Sub optimizer steps = 40 / 40
Main policy loss = 0.2346
Sub policy loss = 0.2073
Main KL ~= 0
Sub KL = 0.000233
clip fraction:
  Main = 0.0000
  Sub = 0.0016
val after = 0.200
```

aligned validation 拒绝了更新后的 checkpoint，best 仍保留 SFT200。

### 7.5 Strict Main-Only

20-task Main-only 实验在运行过程中被人工中断。

中断后残留进程已停止，partial 输出目录保留。该实验没有完整结果，不能用于结论。

## 8. 综合结果表

| 阶段 | 训练方法 | Eval | Success | Invalid | 结论 |
|---|---|---|---:|---:|---|
| SFT50 | action-only | easy5 | 0.400 | 0.280 | 能解题但格式不稳 |
| SFT50 | structured | easy5 | 0.200 | 0.040 | 格式改善，数据不足 |
| SFT200 | structured + short history | easy100 | **0.400** | 0.097 | 当前可靠基线 |
| GRPO-style | per-step update | easy20 | 0.450 | 0.083 | 低于 SFT |
| GRPO-style | batch + replay | easy20 | 0.550 | 0.034 | 小切片正收益 |
| GRPO-style | batch + replay | easy100 | 0.380 | **0.078** | 正收益未复现 |
| GRPO-style | independent 50-task | easy100 | 0.380 | 0.092 | 扩大训练仍未提升 |
| Strict GRPO | joint, 20-task | aligned easy20 | 0.200 | 0.048 | 从 0.550 明显下降 |

## 9. 当前结论

### 9.1 已经成立

1. Plancraft 比数学题更适合当前实验：
   - 环境可执行；
   - reward 可验证；
   - 有 oracle；
   - 能观察状态推进和工具动作。
2. Structured Sub 接口有价值：
   - 非法动作率明显降低；
   - Sub 能表达局部目标和理由；
   - SFT200 能在 easy100 解决 40 个任务。
3. Short history 和完整 target token 保留是关键工程改进。
4. 评测必须与 checkpoint selection 使用相同协议。
5. 早期 weighted-SFT 不是严格 GRPO。
6. strict policy-ratio GRPO 已经实现并通过数值及 smoke 测试。

### 9.2 尚未成立

当前没有证据支持：

```text
GRPO 在 Plancraft 上稳定超过 structured SFT200
```

严格 joint GRPO 首轮实验反而降低 aligned validation。

### 9.3 当前瓶颈判断

从最新 strict GRPO 日志看：

```text
ratio 接近 1
clip fraction 接近 0
KL 很小
validation 仍明显下降
```

这意味着问题不像是 policy clipping 或 KL 爆炸，更可能是：

1. rollout reward/advantage 与 held-out 成功率不一致；
2. 对失败轨迹中所有 token 使用同一 episode advantage，credit assignment 仍太粗；
3. Sub/Main agreement reward 鼓励两者复制，而非提供互补信息；
4. oracle progress 能奖励局部推进，但不保证最终可完成；
5. 20-task rollout 分布仍不足以支持泛化更新。

## 10. 下一阶段建议

优先级从高到低：

1. 完成 strict Main-only 5-task 快速对照，不直接重跑数小时的 20-task。
2. 加 step-level advantage：
   - 每步 progress delta；
   - invalid/no-op penalty；
   - terminal success 单独回传；
   - 不再让整条轨迹的所有 token共享完全相同 advantage。
3. 去掉或降低 Sub/Main agreement reward，避免 Sub 退化为 Main 镜像。
4. 增加 reward component 与 task-level failure trace 输出。
5. strict GRPO 在 5-task aligned validation 不下降后，再扩大到 20/50 tasks。
6. 最终只用 easy100 和多 seed 判断是否超过 SFT。

## 11. 当前状态

当前推荐 checkpoint：

```text
plancraft_mas_structured_short_sft_200x1/main_agent
plancraft_mas_structured_short_sft_200x1/sub_agent
```

本次报告同步提交 strict GRPO 代码，主要包含：

```text
grpo_v4.py
grpo_plancraft_mas.py
```

已完成：

- strict GRPO smoke 回归；
- 所有现存 GRPO 脚本 py_compile；
- strict joint 20-task aligned validation；
- 中断后台进程清理。

尚未完成：

- strict Main-only 完整对照；
- strict GRPO easy100 评测。由于 aligned easy20 已拒绝更新后的 checkpoint，
  当前没有必要将该 checkpoint 作为最终候选做大规模评测。

## 12. Strict GRPO Failure Analysis And Step Credit

统一 SFT、独立 evaluator 和 GRPO 的 history 协议为最近 3 步后，重新评估：

| Model | Success | Avg steps | Invalid |
|---|---:|---:|---:|
| Structured SFT200 | 0.600 | 3.900 | 0.027 |
| Strict joint step checkpoint | 0.400 | 5.900 | 0.102 |

配对结果：

```text
SFT-only solves = 5
RL-only solves = 1
net regression = 4 / 20
```

主要失败模式：

```text
1. 应使用 crafting grid 时重复 smelt。
2. 应 smelt 时把材料依次填入 crafting grid。
3. slab 等配方使用错误的横向/纵向布局。
4. 可解任务上过早输出 impossible。
5. 执行合法但错误的动作直到 max_steps。
6. Sub/Main 在失败步骤上的动作完全一致，Main 没有纠错。
```

失败轨迹中：

```text
SFT timeout = 2 / 8 failures
strict joint timeout = 7 / 12 failures
Sub/Main exact action agreement = 100%
```

因此问题不只是动作格式。旧 objective 将同一个 episode advantage 分配给轨迹中
每一步，并额外奖励 Sub/Main agreement，容易强化错误动作模板。

已改为 step-level credit：

```text
每步 Main/Sub 独立 reward
terminal success 只奖励成功终止步
oracle progress 奖励实际状态推进步
invalid/no-progress 不能继承最终成功 credit
重复动作单独惩罚
错误 impossible 单独惩罚
Sub/Main agreement 默认权重降为 0
同一 step index 在 group 内计算 relative advantage
```

新增训练诊断：

```text
repeat rate
incorrect-stop rate
unique first-action rate
Main/Sub zero-advantage rate
```

2-task、group-size 4 smoke：

```text
zero-advantage rate = 0.500
Main/Sub optimizer steps = 2 / 2
Main policy loss = 0.0008
Sub policy loss = 0.0016
Sub KL = 0.000180
validation success = 1.000 -> 1.000
```

这验证了新的 step-level strict GRPO 链路可以在有 reward 差异的步骤上更新，同时
跳过没有区分度的步骤。正式实验仍需从同一 SFT checkpoint 运行
Main-only、Sub-only 和 joint 对照。
