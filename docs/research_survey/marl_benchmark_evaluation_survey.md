# Agentic RL 多智能体方向 Benchmark 与 Evaluation 方法调研

## 一、多智能体 RL 标准 Benchmarks

### 1.1 SMAC / SMACv2 (StarCraft Multi-Agent Challenge)
- **SMAC**: 基于 StarCraft II 的协作多智能体基准测试，是 MARL 领域最广泛使用的 benchmark 之一
- **核心特点**: 部分可观测、异构智能体、需要协作策略
- **评估指标**: 
  - Win rate（胜率）
  - Mean reward / Episode return
  - Dead units（死亡单位数，衡量效率）
- **SMACv2 (2023)**: 改进版本，增加了更强的随机性和更复杂的场景
  - 引入 entity-based 状态表示
  - 更难的探索挑战

### 1.2 MPE (Multi-Agent Particle Environment)
- **来源**: OpenAI 开发，后被广泛使用
- **核心特点**: 2D 粒子世界，轻量级，易于扩展
- **场景类型**:
  - Cooperative navigation
  - Predator-prey
  - Communication
  - Covert communication
- **评估指标**: 平均回报、碰撞次数、目标达成率、通信成功率

### 1.3 Google Research Football
- **来源**: Google Research
- **核心特点**: 基于足球的复杂多智能体环境
- **评估指标**: Goal difference、Win rate、Pass accuracy、Ball possession time

### 1.4 其他重要 Benchmarks
| Benchmark | 特点 | 适用场景 |
|-----------|------|----------|
| **MAMuJoCo** | 连续控制，多智能体机器人 | 协作连续控制 |
| **Hanabi** | 部分可观测卡牌游戏 | 隐式通信、心智理论 |
| **Overcooked** | 需要协调的烹饪游戏 | 人类-AI 协作 |
| **LBF (Level-Based Foraging)** | 层级觅食 | 异构智能体协作 |
| **RWARE (Robot Warehouse)** | 仓库机器人调度 | 大规模 MARL |
| **MAgent** | 大规模群体智能 | 数百到数千智能体 |

---

## 二、LLM-based Multi-Agent 评估方法

### 2.1 核心评估维度

#### 2.1.1 任务完成度 (Task Completion)
- **Success Rate**: 任务成功完成的比例
- **Task-specific metrics**: 如代码正确率、文档质量分数
- **Goal achievement**: 与预设目标的匹配度

#### 2.1.2 协作效率 (Collaboration Efficiency)
- **Communication overhead**: 通信轮次、token 消耗
- **Coordination success**: 协调动作的成功率
- **Role specialization**: 角色分工的明确程度

#### 2.1.3 系统鲁棒性 (System Robustness)
- **Failure recovery**: 失败后的恢复能力
- **Adversarial robustness**: 对抗扰动下的表现
- **Scalability**: 智能体数量增加时的性能变化

### 2.2 现有 Benchmarks 与框架

| 框架/基准 | 描述 | 评估重点 |
|-----------|------|----------|
| **AgentBench** | 多场景 LLM Agent 评估 | 工具使用、推理、决策 |
| **WebArena** | 网页交互任务 | 自主导航、信息检索 |
| **ToolBench** | 工具学习评估 | API 调用、工具组合 |
| **MLAgentBench** | 机器学习研究任务 | 端到端 ML 实验 |
| **MetaGPT** | 软件开发多智能体 | 代码质量、项目完成度 |
| **AutoGen** | 对话式多智能体 | 对话质量、任务完成 |
| **ChatDev** | 虚拟软件公司 | 软件工程指标 |

### 2.3 多智能体特有评估指标

```
1. 集体智能指标:
   - Collective IQ: 群体解决问题的效率
   - Emergent behavior score: 涌现行为量化

2. 通信评估:
   - Communication effectiveness: 信息传递准确率
   - Message redundancy: 消息冗余度
   - Protocol emergence: 通信协议的形成

3. 社会选择指标:
   - Social welfare: 社会福利总和
   - Fairness index: 公平性指数 (Jain's fairness index)
   - Pareto efficiency: 帕累托效率
```

---

## 三、信用分配 (Credit Assignment) 经典方法与最新进展

### 3.1 经典方法

#### 3.1.1 值分解方法 (Value Decomposition)
| 方法 | 核心思想 | 代表工作 |
|------|----------|----------|
| **VDN** | 将联合 Q 值分解为个体 Q 值之和 | Sunehag et al., 2017 |
| **QMIX** | 使用单调性约束的混合网络 | Rashid et al., 2018 |
| **QTRAN** | 可分解性转换，处理非单调情况 | Son et al., 2019 |
| **QPLEX** | duplex dueling 架构 | Wang et al., 2020 |
| **RODE** | 学习角色选项分解 | Wang et al., 2021 |

#### 3.1.2 策略梯度方法
| 方法 | 核心思想 | 代表工作 |
|------|----------|----------|
| **COMA** | 反事实基线进行信用分配 | Foerster et al., 2018 |
| **MAPPO** | 多智能体 PPO，简单但有效 | Yu et al., 2021 |
| **HAPPO** | 顺序更新保证单调改进 | Kuba et al., 2021 |

#### 3.1.3 反事实方法 (Counterfactual)
- **COMA (Counterfactual Multi-Agent Policy Gradients)**:
  - 核心: 计算每个智能体的边际贡献
  - 公式: A^a = Q(s, u) - sum_{u^a} pi^a(u^a|tau^a) Q(s, (u^{-a}, u^a))
  - 通过比较实际动作与平均动作的 Q 值来分配信用

### 3.2 最新进展 (2023-2024)

#### 3.2.1 基于 Shapley 值的信用分配
- **SHAQ (Shapley Q-value)**:
  - 将 Shapley 值引入多智能体信用分配
  - 满足效率性、对称性、虚拟玩家、可加性公理
  - 论文: "Shapley Q-Value: A Local Reward Approach to Solve Global Reward Games"

- **SV-RPG (Shapley Value based Reward Redistribution)**:
  - 将团队奖励按 Shapley 值重新分配
  - 解决非对称贡献的信用分配问题

#### 3.2.2 基于因果推断的信用分配
- **Causal Multi-Agent Credit Assignment**:
  - 使用因果图识别每个智能体的因果效应
  - 区分直接效应和间接效应
  - 论文: "Causal Multi-Agent Credit Assignment" (ICML 2023/2024)

#### 3.2.3 基于注意力机制的信用分配
- **ATOC (Attentional Communication)**:
  - 学习何时通信、与谁通信
  
- **TarMAC (Targeted Multi-Agent Communication)**:
  - 定向注意力通信

#### 3.2.4 基于信息论的信用分配
- **IC3Net (Individualized Controlled Continuous Communication)**:
  - 学习通信的 gated 机制
  
- **MAGIC (Multi-Agent Graph AttentIon Communication)**:
  - 图注意力通信结构

### 3.3 信用分配方法对比

```
评估维度:
1. 可扩展性 (Scalability): 智能体数量增加时的表现
2. 计算效率: 信用分配的计算开销
3. 理论保证: 是否有收敛性/最优性保证
4. 非对称处理: 处理异构智能体的能力
5. 在线学习: 是否支持在线信用分配

方法对比:
- VDN/QMIX: 高可扩展性，但受限于可分解性假设
- COMA: 需要完整联合动作空间，可扩展性有限
- Shapley-based: 理论优美，但计算复杂度高
- Causal: 能区分因果效应，但需要因果模型
```

---

## 四、与用户工作的关联分析

### 4.1 Goal Misalignment Benchmark 相关

#### 4.1.1 目标不对齐的 MARL 研究
| 研究方向 | 相关工作 | 关联度 |
|----------|----------|--------|
| **意图推断 (Intent Inference)** | 从行为推断其他智能体目标 | 高 |
| **对手建模 (Opponent Modeling)** | 建模其他智能体的策略/目标 | 高 |
| **目标条件 RL (Goal-Conditioned RL)** | 学习目标条件策略 | 中 |
| **Safe MARL** | 安全约束下的多智能体学习 | 中 |

#### 4.1.2 相关评估指标
```
1. 目标对齐度度量:
   - Goal divergence: 目标分布的 KL 散度
   - Intent alignment score: 意图对齐分数
   - Value alignment: 价值函数一致性

2. 行为层面的不对齐检测:
   - Action inconsistency: 动作不一致率
   - Sub-goal conflict: 子目标冲突检测
   - Reward hacking detection: 奖励篡改检测
```

### 4.2 BudgetController (Credit-guided Test-time Scaling) 相关

#### 4.2.1 测试时计算分配的相关工作
| 方向 | 描述 | 代表工作 |
|------|------|----------|
| **Adaptive Computation Time** | 根据输入动态调整计算 | Graves, 2016 |
| **Early Exit Networks** | 提前退出节省计算 | Teerapittayanon et al., 2016 |
| **Mixture of Depths** | 动态选择层深度 | Rae et al., 2024 |
| **Best-of-N Sampling** | 采样多个选择最优 | LLM 推理常用 |
| **Tree of Thoughts** | 思维树搜索 | Yao et al., 2023 |

#### 4.2.2 多智能体场景的计算分配
```
核心问题: 如何在多个智能体之间分配有限的计算预算

相关概念:
1. 资源分配博弈 (Resource Allocation Games):
   - 将计算预算视为资源
   - 使用博弈论方法分配

2. 优先级经验回放 (Prioritized Experience Replay):
   - 根据重要性分配更新资源
   - 可扩展到测试时计算分配

3. 元学习/学习优化 (Learning to Optimize):
   - 学习如何分配计算
   - MAML-style 的预算分配
```

#### 4.2.3 与信用分配的直接关联
```
BudgetController 与 Credit Assignment 的深层联系:

1. 计算信用的概念:
   - 将"计算资源"视为一种需要分配的信用
   - 每个智能体/子任务对最终目标的贡献 = 应分配的计算量

2. 反事实推理的应用:
   - 估计"如果给某个智能体更多/更少计算，结果会如何"
   - 类似 COMA 的反事实基线

3. 边际效用分析:
   - 计算分配的边际效用递减点
   - 最优预算分配 = 各智能体边际效用相等
```

### 4.3 可直接借鉴的评估方法

#### 4.3.1 对于 Goal Misalignment Benchmark
```
建议采用的 MARL 评估技术:
1. 使用反事实评估:
   - 测量"如果子智能体与父智能体目标一致，性能提升多少"
   - 量化不对齐造成的损失

2. Shapley 值分析:
   - 评估每个子智能体对父目标的贡献
   - 识别"搭便车"或"有害"的子智能体

3. 因果效应估计:
   - 使用 do-calculus 估计目标不对齐的因果效应
   - 区分相关性与因果性
```

#### 4.3.2 对于 BudgetController
```
建议采用的评估指标:
1. 计算效率指标:
   - Compute-normalized performance: 归一化计算后的性能
   - Performance per FLOP: 每浮点运算的性能
   - Budget utilization efficiency: 预算使用效率

2. 分配质量指标:
   - Allocation fairness: 分配公平性 (Jain's index)
   - Pareto optimality: 帕累托最优性检验
   - Regret analysis: 与最优分配的后悔值

3. 动态调整指标:
   - Adaptation speed: 对任务变化的适应速度
   - Stability: 分配策略的稳定性
```

---

## 五、关键论文与资源

### 5.1 必读论文清单

**信用分配经典:**
1. Rashid et al. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning" (ICML 2018)
2. Foerster et al. "Counterfactual Multi-Agent Policy Gradients" (AAAI 2018)
3. Sunehag et al. "Value-Decomposition Networks For Cooperative Multi-Agent Learning" (AAMAS 2017)

**信用分配最新:**
4. Li et al. "Shapley Q-Value: A Local Reward Approach to Solve Global Reward Games" (AAAI 2021)
5. "Causal Multi-Agent Credit Assignment" (ICML 2023/2024)
6. Wang et al. "QPLEX: Duplex Dueling Multi-Agent Q-Learning" (ICLR 2021)

**LLM Multi-Agent:**
7. "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework" (2023)
8. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)
9. "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society" (2023)

**评估与Benchmark:**
10. Samvelyan et al. "The StarCraft Multi-Agent Challenge" (2019)
11. Ellis et al. "SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning" (2023)
12. "AgentBench: Evaluating LLMs as Agents" (2023)

### 5.2 开源资源

| 资源 | 链接 | 用途 |
|------|------|------|
| **SMAC** | https://github.com/oxwhirl/smac | MARL 基准测试 |
| **SMACv2** | https://github.com/oxwhirl/smacv2 | 改进版 MARL 基准 |
| **MPE** | https://github.com/openai/multiagent-particle-envs | 轻量级 MARL |
| **EPyMARL** | https://github.com/uoe-agents/epymarl | MARL 算法统一框架 |
| **MAPPO** | https://github.com/marlbenchmark/on-policy | 策略梯度基准 |
| **MetaGPT** | https://github.com/geekan/MetaGPT | LLM 多智能体框架 |

---

## 六、总结与建议

### 6.1 对用户工作的直接建议

**Goal Misalignment Benchmark:**
- 可借鉴 COMA 的反事实基线方法，量化"目标对齐"vs"不对齐"的性能差距
- 考虑使用 Shapley 值来量化每个子智能体对父目标的边际贡献
- 引入因果推断框架，区分"目标不对齐"的因果效应与混杂因素

**BudgetController:**
- 将计算预算分配形式化为资源分配博弈，借鉴博弈论中的分配机制
- 使用信用分配中的"边际贡献"概念，指导计算资源的边际效用分析
- 考虑开发类似 QMIX 的"计算分配分解"框架，将全局预算约束分解为个体约束

### 6.2 潜在研究方向

1. **Credit-Guided Budget Allocation**: 将信用分配理论直接应用于测试时计算分配
2. **Multi-Agent Test-Time Scaling**: 多智能体场景下的测试时扩展策略
3. **Goal-Aware Credit Assignment**: 考虑目标不对齐的信用分配方法
4. **Dynamic Budget-Credit Tradeoff**: 预算与信用之间的动态权衡机制
