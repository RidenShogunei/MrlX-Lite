# 2026-06-02 至 2026-06-08 多智能体训练实验分析报告

> 本报告包含 6 月 2 日的前置工作。若需要“上周三至今”的精确范围，请以
> `EXPERIMENT_REPORT_2026-06-03_TO_2026-06-08.md` 为准。

## 1. 报告范围

本报告汇总北京时间 2026 年 6 月 2 日（上周二）至 6 月 8 日完成的实验、代码改造和结论修正，覆盖：

- Enhanced HotpotQA 环境与固定 MAS；
- 动态 MAS 的 SFT、回放、综合和 verifier 实验；
- Qwen3.5-9B 小规模基线；
- 训练算法审计与 advantage-based 更新；
- Plancraft 环境、Oracle SFT、固定 MAS 和严格 GRPO；
- 当前可信结论、失败原因和下一阶段建议。

这段时间共形成 29 个提交。实验主线从“继续调 HotpotQA reward”逐步转为“把环境、评测和优化器拆开验证”，最终确认：当前最大问题已经不是缺少一次新的 reward 调参，而是训练协议、信用分配和策略优化本身尚未形成稳定闭环。

---

## 2. 执行摘要

### 2.1 已经取得的成果

1. **在 Enhanced HotpotQA 上首次观察到固定 MAS 超过 direct Main。**

   Enhanced Sub preference training 将 Sub oracle reward 从 `0.375` 提升到 `0.508`。冻结该 Sub 并与 Stage2 Main 组合后，多 offset 平均 reward 达到 `0.480`，高于 direct Main 的 `0.463`。

2. **动态 MAS 已经能生成多个 Sub 任务，但还没有形成稳定优势。**

   动态 Main 的平均子任务数约为 `1.8`，说明“动态分派”这一行为可以通过 SFT 学到；但当前几乎不会选择 direct，且最终 answer synthesis 弱于固定 MAS。

3. **定位了动态 MAS 中不同阶段的独立瓶颈。**

   - document catalog 修复了盲猜文档 ID 的规划问题；
   - mixture SFT 将 evidence 提高到 `0.675`；
   - synthesis SFT 将 answer F1 从 `0.224` 提高到 `0.316`；
   - verifier SFT 改善了难切片和 best-of 指标，但降低了平均单样本表现。

4. **完成了 Plancraft 的可执行 RL 环境和可靠 SFT 基线。**

   Oracle sanity check 达到 100%；结构化 SFT200 在 easy100 上成功率 `0.40`，是当前最可信的 Plancraft checkpoint。

5. **发现并修正了此前“GRPO”实现的关键问题。**

   旧 HotpotQA/Plancraft 更新本质上更接近 best-of-N reward-filtered weighted SFT，没有 old-policy ratio、ratio clipping 和 KL 约束。现在已经实现具备 frozen reference、old log-prob、clipped ratio、KL 和多 policy epoch 的严格版本。

### 2.2 尚未解决的问题

1. **严格 GRPO 尚未带来提升。**

   Plancraft strict joint GRPO 在 easy20 上从 SFT 的 `0.55` 降到 `0.20`。这证明训练确实发生了，但优化方向或信用分配仍不正确。

2. **Plancraft 当前不是动态 MAS。**

   当前全部 Plancraft 训练均为固定 `1 Main + 1 Sub`。动态的 `0 / 1 / N Sub` 只在 HotpotQA 上做过 SFT 和评测，没有完成严格 RL 验证。

3. **联合训练仍不如分阶段训练稳定。**

   在 HotpotQA 中，更新 Sub 容易破坏 retrieval、summary 或 Main 所依赖的输出分布；在 Plancraft 中，联合 strict GRPO 同样发生明显退化。

### 2.3 当前最重要的判断

> 我们已经证明 MAS 在合适环境和合适 Sub 能力下可以有价值，但还没有证明当前联合 GRPO 能稳定训练出这种价值。

当前成果更接近一套经过审计的实验平台和一组明确的负结果，而不是已经成功复现 M-GRPO。

---

## 3. 实验时间线

| 日期 | 阶段 | 主要工作 | 结果 |
|---|---|---|---|
| 6 月 2 日 | Enhanced HotpotQA | 扩大文档干扰，建立动态 MAS 原型与 dynamic SFT | 固定 MAS 的价值开始显现；动态分派可被 SFT 学习 |
| 6 月 3 日 | 动态 MAS / 工程清理 | mixture replay、synthesis 数据、删除原 MrlX 代码、Qwen3.5-9B 尝试、TRL GRPO 脚本 | 仓库收敛到自研代码；9B 小样本表现较强但证据不足 |
| 6 月 4 日 | 动态诊断 / Plancraft | verifier、Sub replay、advantage objective、Plancraft 接入与 SFT/GRPO 初版 | 暴露旧 GRPO 实为 weighted SFT；新 benchmark 可执行闭环建立 |
| 6 月 5 日 | Plancraft 扩展 | structured Sub、SFT200、progress reward、大规模 weighted 更新 | SFT 基线稳定；RL 小样本收益未能泛化 |
| 6 月 6 日 | 评测对齐 | independent 50-task rollout、aligned validation | 纠正 best-of-N 和温度不一致造成的虚假增益 |
| 6 月 8 日 | 严格 GRPO | low-strength 消融、policy-ratio GRPO、阶段报告 | strict joint 确实更新但显著退化 |

---

## 4. 阶段一：Enhanced HotpotQA 与固定 MAS

### 4.1 为什么增强环境

原 HotpotQA 本地环境每题约 10 篇文档，direct Main 可以自行 search/read，导致多智能体分工没有足够必要性。Enhanced 环境扩展到约 30 篇候选文档，并扩大训练/验证规模：

- train：500 tasks；
- validation：150 tasks；
- 每题约 30 docs。

目的不是单纯增加难度，而是提高 retrieval 压力，让 Sub 的专门检索能力真正可能产生边际价值。

### 4.2 Sub retrieval 修复

旧 Preference/Stage2 Sub 在 enhanced 环境下的 oracle 结果：

| Sub | support recall | answer F1 | reward |
|---|---:|---:|---:|
| 旧 Preference Sub | 0.378 | 0.285 | 0.375 |
| Joint Sub | 0.355 | 0.275 | 0.364 |

这表明旧 Sub 在更多 distractor 下 retrieval 明显不足，联合更新还会继续损伤检索。

使用 300 个 enhanced train tasks、900 个 preference pairs 训练后：

| Sub | support recall | answer F1 | reward | best reward |
|---|---:|---:|---:|---:|
| 旧 Preference Sub | 0.378 | 0.285 | 0.375 | 0.430 |
| Enhanced Preference Sub | 0.502 | 0.440 | 0.508 | 0.566 |

这是一个较强的因果证据：环境变化后，真正的瓶颈首先是 Sub retrieval，而不是 Main 是否“愿意调用 Sub”。

### 4.3 固定 MAS 首次超过 direct Main

使用 Stage2 Main 与 Enhanced Preference Sub，在 5 个 offset、共 100 个任务上评估：

| 模型 | answer F1 | evidence | reward | best answer F1 | best reward |
|---|---:|---:|---:|---:|---:|
| Direct Main | 0.372 | 0.515 | 0.463 | 0.410 | 0.491 |
| Fixed MAS | 0.398 | 0.505 | 0.480 | 0.496 | 0.556 |

相对提升：

- answer F1：`+0.026`
- reward：`+0.017`
- best answer F1：`+0.086`
- best reward：`+0.065`

### 4.4 这项结果说明什么

该结果支持三个判断：

1. MAS 是否有优势高度依赖环境是否需要真实分工；
2. 强 Sub 不是“多一个模型”这么简单，而是必须在目标环境中具备专门能力；
3. staged training 比 joint training 更稳定，因为它避免同时扰动 Main 和 Sub 的接口分布。

但需要保留一条限制：当时的 direct “GRPO”和部分 MAS 更新使用的是 best-of/reward-filtered 训练，因此这组结果能证明**固定 MAS 组合优于 direct checkpoint**，不能证明论文意义上的 M-GRPO 已经成功。

---

## 5. 阶段二：HotpotQA 动态 MAS

### 5.1 动态架构定义

动态 Main 被训练为输出：

```text
[mode]direct[/mode]
```

或：

```text
[mode]delegate[/mode]
[subtask]...[/subtask]
[subtask]...[/subtask]
```

多个 Sub instance 共享同一个 Sub LoRA，但拥有独立 history。设计目标是支持：

- 不分 Sub；
- 分配 1 个 Sub；
- 分配多个 Sub。

### 5.2 初始 dynamic SFT

500 个 enhanced tasks、3000 条 SFT 样本后：

| 模型 | direct rate | avg subtasks | answer F1 | evidence | reward |
|---|---:|---:|---:|---:|---:|
| staged best，fallback | 0.000 | 1.000 | 0.315 | 0.400 | 0.400 |
| joint dynamic SFT | 0.000 | 1.850 | 0.347 | 0.475 | 0.438 |
| Main-only dynamic SFT | 0.000 | 1.825 | 0.293 | 0.487 | 0.402 |

结论：

- Main 已经学会生成多个 subtasks；
- joint SFT 比 Main-only 更适应新的 focused subtask 分布；
- `direct_rate=0` 表明它没有学会真正的按需路由，只学会了“通常分两个”。

### 5.3 动态 mixture 与 document catalog

早期计划 prompt 不包含文档目录，但要求 Main 输出具体文档 ID，导致 Main 只能猜测。加入 `Dxx: title` catalog 后，dynamic mixture v3 达到：

| 指标 | 结果 |
|---|---:|
| avg subtasks | 1.800 |
| answer F1 | 0.224 |
| evidence | 0.675 |
| reward | 0.392 |

这次结果非常关键：规划和 evidence selection 已部分成功，但最终回答反而成为主要瓶颈。

### 5.4 Main-only synthesis SFT

冻结 Sub，只训练 Main 将多个 focused Sub results 综合成一个最终答案：

| 模型 | answer F1 | evidence | reward |
|---|---:|---:|---:|
| mixture v3 | 0.224 | 0.675 | 0.392 |
| synthesis 300 | 0.254 | 0.662 | 0.410 |
| synthesis 500 | 0.316 | 0.700 | 0.461 |

5 个 offset 的较大评估：

| 模型 | answer F1 | evidence | reward |
|---|---:|---:|---:|
| dynamic mixture | 0.318 | 0.675 | 0.458 |
| dynamic synthesis | 0.344 | 0.655 | 0.472 |
| fixed staged best | 0.413 | 0.495 | 0.488 |

动态 MAS 已能召回更多 evidence，但还不能像固定 MAS 一样稳定利用这些 evidence。

### 5.5 Sub evidence replay 的失败

针对 offset 40 训练 Sub evidence summary：

- 纯 summary continuation 导致灾难性遗忘，reward 降至 `0.257`；
- 加入 action replay 后，offset 40 reward 从 `0.331` 恢复到 `0.374`；
- 但多 offset 平均 answer F1 仅 `0.088`，reward `0.300`。

这说明 Sub 的 action、retrieval、summary 和 answer clue 不是一个简单 scalar reward 能同时维持的能力集合。修复一个局部切片会改变整个 Sub 输出分布。

### 5.6 Main verifier 的收益与代价

Main verifier SFT 在 offset 40 上显著改善：

| Main | answer F1 | reward | best answer F1 | best reward |
|---|---:|---:|---:|---:|
| synthesis | 0.137 | 0.331 | 0.140 | 0.338 |
| verifier | 0.312 | 0.454 | 0.500 | 0.600 |

但多 offset 平均结果出现权衡：

- answer F1：`0.344 -> 0.330`
- reward：`0.472 -> 0.450`
- best answer F1：`0.394 -> 0.455`
- best reward：`0.516 -> 0.550`

它学到了有用但不稳定的 verifier mode，更适合做 reranking 或 distillation，而不是直接替换 synthesis Main。

### 5.7 动态 MAS 阶段结论

动态能力并非完全失败：

- 多 Sub 生成已经学会；
- evidence recall 一度超过固定 MAS；
- targeted synthesis 能持续改善回答。

但它仍不是成熟的动态系统：

- 几乎不会 direct；
- 子任务数量集中在约 1.8，没有真正按任务难度变化；
- 多 Sub evidence 到 final answer 的信用分配不稳定；
- 尚未进行严格 GRPO。

因此目前最准确的命名是“dynamic MAS SFT prototype”，不是“dynamic M-GRPO result”。

---

## 6. 工程清理与模型扩展

### 6.1 仓库清理

6 月 3 日完成两轮清理：

- 整理 HotpotQA MAS 项目结构；
- 删除原作者 `MrlX-DeepResearch`、`MrlX-SelfRewarding`、`MrlX-TakesTwo` 等不再使用的代码；
- 保留当前实验所需的 `lightweight_cotrain` 主线。

这一步的意义是减少“代码存在但实际未运行”的误导，使后续报告能对应真实脚本和 checkpoint。

### 6.2 Qwen3.5-9B 小样本实验

10 个任务、每题 2 samples 的结果：

| 模型 | tool valid | answer F1 | evidence | reward |
|---|---:|---:|---:|---:|
| Qwen3.5-9B base | 0.900 | 0.200 | 0.700 | 0.370 |
| Qwen3.5-9B SFT | 1.000 | 0.542 | 0.650 | 0.609 |
| Qwen3.5-9B dynamic MAS | 0.850 | 0.458 | 0.725 | 0.551 |

这些结果说明大模型 SFT 潜力较强，但样本仅 10 题，且未完成多 offset 与等预算对比，证据等级较低。后续 Plancraft 仍主要使用 Qwen2.5-1.5B，以便快速做训练算法审计。

---

## 7. 训练算法审计

### 7.1 旧实现的问题

审计发现旧循环主要执行：

1. 同一任务采样多个 candidates；
2. 选择 reward 最好的 candidate；
3. 对 winner 做加权 teacher-forcing；
4. 丢弃其余 candidate。

这种方法可称为 best-of-N reward-filtered SFT 或 weighted SFT，但缺少：

- old policy log-prob；
- current/old probability ratio；
- clipped surrogate objective；
- frozen reference KL；
- 对负 advantage 样本的显式抑制。

因此此前“GRPO 稳定超过 SFT”的说法必须降级。那些实验仍能说明优质轨迹回放可能有效，但不能作为真正 RL 提升的证据。

### 7.2 Advantage-based 中间版本

随后加入 group-relative advantage：

- 组内标准化 final reward；
- 正 advantage 提高轨迹概率；
- 负 advantage 降低轨迹概率；
- Main/Sub 使用各自训练 reward。

它修复了“坏样本完全被丢弃”的问题，但仍没有 ratio clipping 和 KL，因此只是向 GRPO 迈进的中间版本。

### 7.3 TRL 路线

曾接入 `GRPOTrainer` 并修正 reward kwargs，使 reward 能读取 dataset 中的 `gold_answer` 和 `gold_doc_ids`。但现有 MAS 是多轮、双 adapter、环境交互式轨迹，直接套单响应 trainer 难以表达 Main/Sub 分离信用，因此没有成为最终主线。

---

## 8. 阶段三：Plancraft 环境与 SFT 基线

### 8.1 为什么换 Plancraft

HotpotQA 的开放式检索和答案生成使 reward 同时承担 retrieval、evidence、summary、answer correctness 和 routing 多种目标，难以判断训练失败究竟来自模型还是评测。

Plancraft 提供：

- 可执行 crafting 状态；
- 确定性的合法动作；
- Oracle plan；
- 每一步可计算的 state progress；
- 明确的 success 和 invalid action 指标。

它更适合先验证“RL 优化器能否真正提升策略”，再扩展到动态 MAS。

### 8.2 当前 Plancraft 架构

当前是固定：

```text
Main 选择或规划下一步
  -> Sub 生成一个结构化 crafting action
  -> Environment 执行动作并返回新状态
  -> 重复直到成功或达到步数上限
```

这里始终是 `1 Main + 1 Sub`。没有实现：

- Main 选择不调用 Sub；
- 一步调用多个 Sub；
- 多个 Sub 并行或递归。

### 8.3 Oracle 与 SFT

Oracle sanity check 在小样本上达到 100%，证明环境和任务本身可解。

SFT 结果：

| SFT 版本 | 评估 | success | avg steps | invalid rate |
|---|---|---:|---:|---:|
| action-only SFT50 | easy5 | 0.40 | 6.8 | 0.28 |
| structured SFT50 | easy5 | 0.20 | 8.4 | 0.04 |
| structured short-history SFT200 | easy5 | 0.40 | 4.6 | 0.07 |
| structured short-history SFT200 | easy100 | 0.40 | 5.57 | 0.097 |

结构化接口显著降低 invalid action，扩大到 200 tasks 后恢复成功率并提高效率。因此当前最佳可靠 checkpoint 是 structured SFT200。

---

## 9. Plancraft Reward 与 weighted-SFT 实验

### 9.1 Reward 分解

Main reward 包括：

- episode success；
- action validity；
- oracle action match；
- oracle progress；
- step efficiency。

Sub reward 包括：

- action validity；
- oracle match；
- state progress；
- Main/Sub agreement；
- 局部执行贡献。

这种拆分比 Main/Sub 共用最终 reward 更合理，但它仍只是手工 credit assignment，不能保证局部最优与最终成功一致。

### 9.2 初期看似提升

小规模 weighted 更新曾出现：

- easy20：SFT `0.50`，batch+replay `0.55`。

但扩大到 easy100 后：

| 模型 | success |
|---|---:|
| SFT | 0.40 |
| weighted update | 0.38 |

配对统计：

- GRPO-only 成功：7；
- SFT-only 成功：9；
- success difference：`-0.02`；
- bootstrap 95% CI：`[-0.10, 0.06]`；
- McNemar `p=0.804`。

结论：小样本上的 `+0.05` 不具备统计稳定性。

### 9.3 独立 50-task rollout

使用 offset 200 的 50 个训练任务、200 条 rollout 后，easy100：

- success：`0.38`；
- efficiency：`0.162`；
- avg steps：`5.85`；
- invalid rate：`0.092`。

扩大 rollout 数据没有自然带来泛化提升。

---

## 10. 评测协议对齐

早期训练和评测存在以下不一致：

- SFT 与更新后模型使用不同 temperature；
- best-of-N 指标与单样本指标混用；
- rollout tasks 与 validation tasks 的难度和 offset 不完全对齐；
- 小样本变化被过度解释。

对齐到相同 decoding 和 easy20 后：

| 模型 | success |
|---|---:|
| SFT baseline | 0.55 |
| 旧 weighted update | 0.25 |
| low-strength Main-only | 0.25 |
| low-strength joint | 0.25 |

这一结果推翻了“只需把学习率调低，weighted 更新就会稳定”的猜测。主要问题不是更新强度，而是目标本身不是可靠的 policy objective。

---

## 11. 严格 Plancraft GRPO

### 11.1 新实现

严格版本加入：

- rollout 时保存 old log-prob；
- frozen Main/Sub reference adapter；
- current/old policy ratio；
- clipped surrogate objective；
- explicit KL penalty；
- 两个 policy epochs；
- Main/Sub 独立 advantage 和优化步骤；
- held-out validation 选择 best checkpoint。

这已经具备标准 GRPO/PPO 风格的核心组件，虽然仍是轻量实现，不等同于完整 M-GRPO 论文系统。

### 11.2 Joint strict GRPO 结果

20-task strict joint：

| 指标 | 结果 |
|---|---:|
| 初始 easy20 success | 0.55 |
| 训练后 easy20 success | 0.20 |
| train success | 0.25 |
| Main optimizer steps | 40 / 40 |
| Sub optimizer steps | 40 / 40 |
| Main policy loss | 0.2346 |
| Sub policy loss | 0.2073 |
| Main KL | 约 0 |
| Sub KL | 0.000233 |
| Main clip fraction | 0 |
| Sub clip fraction | 0.0016 |

### 11.3 如何解释

这不是“代码没训练”：

- optimizer step 确实发生；
- policy loss 非零；
- Sub 出现非零 KL 和 clipping。

更可能的问题是：

1. episode success 太稀疏，局部 shaping reward 主导更新；
2. Main/Sub advantage 仍不能准确反映各自的反事实贡献；
3. rollout batch 太小，组内 advantage 方差大；
4. Main 与 Sub 同时变化，使接口分布漂移；
5. KL 很小但性能大幅下降，说明模型可能在少量关键 token 上发生高影响变化；
6. 当前 reward 可以鼓励“看起来接近 Oracle 的动作”，但未必鼓励完整计划可达性。

严格 Main-only 对照曾启动，但因运行时间较长被中止，当前没有可报告结论。因此还不能区分退化主要来自 Main 更新、Sub 更新，还是两者耦合。

---

## 12. 跨阶段瓶颈分析

这一周实验最有价值的地方，是瓶颈被逐层剥离：

| 阶段 | 表面问题 | 实际定位 |
|---|---|---|
| 原 HotpotQA | MAS 不如 direct | 环境太容易，direct Main 不需要分工 |
| Enhanced HotpotQA | MAS 仍弱 | Sub retrieval 不适应更多 distractors |
| Enhanced Sub 后 | joint 不稳 | 更新扰动 Sub summary 与 Main/Sub 接口 |
| Dynamic MAS | 多 Sub 没提升 | evidence 已提高，但 final synthesis 不稳定 |
| Sub summary replay | 某切片修复、整体崩溃 | 多种 Sub 能力无法由单一 continuation 同时保持 |
| Plancraft weighted RL | 小样本提升 | 评测不对齐且算法实为 weighted SFT |
| Strict GRPO | 真实更新但性能下降 | reward credit、batch 方差和联合非平稳性成为主瓶颈 |

因此用户此前的判断是正确的：

> 不同任务有不同瓶颈，一种固定 reward 很难覆盖所有阶段。

但更进一步说，问题不只是“需要更复杂的复合 reward”。复合 reward 仍可能把互相冲突的目标压成一个 scalar。更合理的方向是：

- 按角色和阶段定义局部可验证目标；
- 保留最终任务成功作为主目标；
- 使用反事实或 leave-one-agent-out credit；
- 对不同能力使用 replay/约束，避免联合更新破坏已有技能；
- 始终用同一 held-out protocol 比较 SFT 和 RL。

---

## 13. 当前结论与证据等级

### 高可信

1. Plancraft structured SFT200 是当前最可靠基线，easy100 success 为 `0.40`。
2. 旧训练循环不是严格 GRPO，而是 reward-filtered/weighted SFT。
3. aligned validation 下，旧 weighted 更新和 low-strength 更新均明显低于 SFT。
4. strict joint GRPO 确实进行了策略更新，但 validation 从 `0.55` 降到 `0.20`。
5. 当前 Plancraft 架构固定为 `1 Main + 1 Sub`，不是动态 MAS。

### 中等可信

1. Enhanced HotpotQA 中，强 retrieval Sub 能让固定 MAS 超过 direct Main。
2. staged training 比 joint training 更稳定。
3. 动态 MAS 的主要后期瓶颈是多证据综合，而不是单纯 retrieval。
4. verifier 更适合 selection/distillation，而不是直接替换 Main。

### 低可信或待验证

1. Qwen3.5-9B SFT 明显更强：当前只有 10-task 小样本。
2. strict GRPO 失败主要由 Sub 导致：缺少完成的 Main-only strict 对照。
3. 更大 RL 数据一定会解决问题：现有 independent rollout 已表明单纯扩量不足。
4. 动态 MAS 能通过当前 reward 自动学会合理的 `0 / 1 / N Sub`：尚未做严格动态 RL。

---

## 14. 当前资产与最佳 checkpoint

### HotpotQA

- 固定 MAS：
  - Main：`hotpotqa_mas_stage2_main_prefsub_50x2/best/main`
  - Sub：`hotpotqa_sub_pref_enhanced_300x900/sub`
- 动态证据/综合主线：
  - Main：`hotpotqa_dynamic_synthesis_mainonly_500x1/main_agent`
  - Sub：`hotpotqa_dynamic_mixture_sft_300x1_v3/sub_agent`
- verifier checkpoint 仅适合研究和 reranking，不应直接替代当前 synthesis Main。

### Plancraft

- 当前可靠起点：structured short-history SFT200；
- 旧 weighted-SFT checkpoints 不应被称为 GRPO best；
- strict joint checkpoint 仅用于失败分析，不应作为部署或下一阶段默认初始化。

---

## 15. 下一阶段建议

### 优先级 1：完成严格 GRPO 的最小因果消融

在同一 SFT200、同一 20/100-task validation、相同 decoding 下依次运行：

1. strict Main-only，Sub 冻结；
2. strict Sub-only，Main 冻结；
3. strict joint；
4. no-update rollout control。

这四组能回答退化究竟来自哪个角色，而不是继续猜 reward 权重。

### 优先级 2：把 reward 改为 step transition credit

Plancraft 每一步都有可执行状态，应该优先使用：

- 动作前后 oracle distance 变化；
- 是否新增目标所需物品；
- 是否消耗不可恢复的关键资源；
- 当前动作是否仍保留完成计划的可达性；
- 最终 success。

Main 和 Sub 不应共享同一个 episode scalar。Main 更应对计划方向负责，Sub 更应对动作合法性和状态转移负责。

### 优先级 3：提高每组有效多样性

如果 group 内候选动作高度相同，GRPO advantage 几乎没有信息。训练前应记录：

- unique action rate；
- reward standard deviation；
- zero-advantage group rate；
- action-level entropy；
- positive/negative advantage 数量；
- ratio、KL 和 clip fraction 分布。

只有 group 具备真实差异时，扩大数据量才有意义。

### 优先级 4：严格 RL 稳定后再做动态 Plancraft

动态版本可以扩展为：

```text
Main:
  direct action
  or delegate one/multiple candidate actions

Sub agents:
  propose independently

Main:
  select / aggregate / reject
```

但在固定 `1 Main + 1 Sub` 上仍无法稳定超过 SFT 时，直接增加动态 agent 数量只会扩大非平稳性和信用分配难度。

---

## 16. 最终阶段判断

从 6 月 2 日到 6 月 8 日，我们完成的不是一次简单的“调参冲榜”，而是三次重要的认识修正：

1. **MAS 价值取决于环境是否真正需要分工。**
2. **动态分派可以被 SFT 教会，但多 Agent 数量本身不等于更强能力。**
3. **此前看似有效的 GRPO 主要是优质轨迹回放；严格策略优化尚未超过 SFT。**

所以当前项目的真实进度是：

> 环境、SFT baseline、固定/动态 MAS 原型、评测协议和严格 GRPO 骨架均已建立；固定 MAS 在 Enhanced HotpotQA 上观察到可信优势，但严格联合 RL 尚未成功，动态 MAS 也尚未进入严格 RL 验证阶段。

下一阶段不应继续无序增加实验，而应围绕 Main-only、Sub-only、joint、no-update 四组严格消融，先把“谁被错误更新、哪种 credit 有效”回答清楚。
