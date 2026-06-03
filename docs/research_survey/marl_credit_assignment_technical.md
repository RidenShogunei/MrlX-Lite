# MARL 信用分配技术细节与实现参考

## 一、经典信用分配算法详解

### 1.1 VDN (Value Decomposition Networks)

**核心公式:**
Q_tot(s, u) = sum_i Q_i(s, u_i)

**特点:**
- 假设联合 Q 值可完全分解为个体 Q 值之和
- 实现简单，训练稳定
- 局限性: 无法表示非线性交互

**PyTorch 伪代码:**
```python
class VDNMixer(nn.Module):
    def forward(self, q_values, states=None):
        # q_values: [batch, n_agents]
        return q_values.sum(dim=-1)  # [batch]
```

### 1.2 QMIX

**核心公式:**
Q_tot(s, u) = f_s(Q_1(s, u_1), Q_2(s, u_2), ..., Q_n(s, u_n))

**约束条件 (单调性):**
partial Q_tot / partial Q_i >= 0, for all i

**实现架构:**
- 使用 hypernetwork 生成混合网络的权重
- 保证非负权重以满足单调性约束

**PyTorch 伪代码:**
```python
class QMIXMixer(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Hypernetworks for weights and biases
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, q_values, states):
        # q_values: [batch, n_agents]
        # states: [batch, state_dim]
        batch_size = q_values.size(0)
        
        # First layer
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)
        b1 = self.hyper_b1(states)
        
        hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1).squeeze(1) + b1)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        b2 = self.hyper_b2(states)
        
        q_tot = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        return q_tot
```

### 1.3 COMA (Counterfactual Multi-Agent Policy Gradients)

**核心思想:**
使用反事实基线来评估每个智能体的边际贡献

**优势函数:**
A^a(s, u) = Q(s, u) - sum_{u^a} pi^a(u^a|tau^a) Q(s, (u^{-a}, u^a))

**实现要点:**
1. 需要联合动作 Q 值网络 Q(s, u)
2. 需要计算反事实基线（对所有可能动作的期望）
3. 使用 critic 网络估计状态-动作值

**与 BudgetController 的关联:**
- COMA 的反事实推理可直接用于计算分配
- "如果给智能体 a 更多预算，Q 值会如何变化"

## 二、Shapley 值在信用分配中的应用

### 2.1 Shapley 值基础

**定义:**
phi_i(v) = sum_{S subseteq N \{i\}} (|S|!(|N|-|S|-1)!)/|N|! [v(S union {i}) - v(S)]

**性质:**
1. **效率性**: sum_{i in N} phi_i(v) = v(N)
2. **对称性**: 对称玩家获得相同 Shapley 值
3. **虚拟玩家**: 无贡献者获得 0
4. **可加性**: 可加游戏的 Shapley 值可加

### 2.2 SHAQ (Shapley Q-Value)

**核心创新:**
将 Shapley 值作为局部奖励，解决全局奖励游戏的信用分配

**计算近似:**
由于精确 Shapley 值计算复杂度为 O(n!)，使用采样近似:
phi_hat_i approx (1/m) sum_{j=1}^{m} [v(S_j union {i}) - v(S_j)]

其中 S_j 是从所有子集中均匀采样

**与 BudgetController 的关联:**
- Shapley 值可量化每个子智能体/子任务的边际贡献
- 可用于指导预算分配: 高 Shapley 值 = 高预算优先级

## 三、因果推断信用分配

### 3.1 因果图模型

**基本框架:**
U_i (智能体i的动作) -> Y (团队奖励)
  ^
S (环境状态)

**因果效应估计:**
- **Total Effect (TE)**: E[Y | do(U_i=u_i)] - E[Y | do(U_i=u_i')]
- **Direct Effect**: 排除其他智能体中介效应后的直接因果效应
- **Indirect Effect**: 通过影响其他智能体产生的间接效应

### 3.2 在 Goal Misalignment 中的应用

**场景:**
父智能体 P 和子智能体 C，可能存在目标不对齐

**因果问题:**
P(团队成功 | do(C 对齐 P 的目标)) = ?

**识别策略:**
1. 使用后门准则控制混杂变量
2. 使用工具变量处理未观测混杂
3. 使用 do-calculus 进行因果效应识别

## 四、计算预算分配的博弈论视角

### 4.1 资源分配博弈建模

**玩家**: N = {1, 2, ..., n} (智能体/子任务)

**策略**: x_i in [0, B] (分配的计算预算)

**约束**: sum_{i in N} x_i <= B (总预算约束)

**效用函数**: u_i(x_i, x_{-i}) = f_i(x_i) - c(x_i)

其中 f_i 是性能函数，c 是成本函数

### 4.2 与信用分配的对偶关系

**原始问题 (信用分配):**
给定团队奖励 R，分配给各智能体 r_1, r_2, ..., r_n

**对偶问题 (预算分配):**
给定总预算 B，分配给各智能体 b_1, b_2, ..., b_n

**关键洞察:**
- 两者都是资源分配问题
- 信用分配: 分配"过去的贡献"
- 预算分配: 分配"未来的资源"
- 可以使用相同的数学框架

### 4.3 最优预算分配条件

**一阶条件 (纳什均衡):**
partial u_i / partial x_i = 0 => f_i'(x_i*) = c'(x_i*)

**社会最优 (中央规划者):**
max_{x_1, ..., x_n} sum_{i in N} u_i(x_i, x_{-i})
s.t. sum_{i in N} x_i <= B

**与 BudgetController 的关联:**
- BudgetController 可以建模为中央规划者问题
- 使用信用分配的结果来估计 f_i（性能函数）
- 动态调整分配以达到社会最优

## 五、评估指标的具体计算

### 5.1 信用分配质量指标

#### 5.1.1 信用分配准确率 (Credit Assignment Accuracy)

**定义:**
衡量分配的信用与"真实"贡献的匹配程度

**计算:**
Accuracy = 1 - sum_i |c_i - c_i*| / sum_i c_i*

其中 c_i 是分配的信用，c_i* 是真实贡献

#### 5.1.2 激励兼容性 (Incentive Compatibility)

**定义:**
信用分配是否激励智能体采取团队最优行动

**检验:**
E[r_i | a_i = a_i*] >= E[r_i | a_i = a_i'], for all a_i' != a_i*

### 5.2 预算分配效率指标

#### 5.2.1 计算效率 (Computational Efficiency)

eta = P(B) / P_max * B_min / B

#### 5.2.2 帕累托效率检验

检查是否存在帕累托改进

## 六、实现建议

### 6.1 Goal Misalignment Benchmark 的实现

```python
class GoalMisalignmentEvaluator:
    def __init__(self, n_agents, parent_goal, child_goals):
        self.n_agents = n_agents
        self.parent_goal = parent_goal
        self.child_goals = child_goals
    
    def compute_alignment_score(self):
        alignment = []
        for child_goal in self.child_goals:
            score = cosine_similarity(self.parent_goal, child_goal)
            alignment.append(score)
        return np.mean(alignment)
    
    def compute_shapley_contribution(self, agent_id, episodes):
        contributions = []
        for _ in range(self.n_samples):
            coalition = self.sample_coalition(agent_id)
            with_agent = self.simulate(coalition + [agent_id])
            without_agent = self.simulate(coalition)
            contributions.append(with_agent - without_agent)
        return np.mean(contributions)
```

### 6.2 BudgetController 的实现

```python
class CreditGuidedBudgetController:
    def __init__(self, n_agents, total_budget, credit_estimator):
        self.n_agents = n_agents
        self.total_budget = total_budget
        self.credit_estimator = credit_estimator
    
    def allocate_budget(self, state, task_complexity):
        credits = self.credit_estimator.estimate(state)
        marginal_utilities = []
        for i in range(self.n_agents):
            mu = self.compute_marginal_utility(i, credits[i], task_complexity)
            marginal_utilities.append(mu)
        allocation = self.solve_optimal_allocation(marginal_utilities, self.total_budget)
        return allocation
```

## 七、相关数学工具

### 7.1 凸优化

预算分配问题通常可表述为凸优化:
min_x -sum_i f_i(x_i)
s.t. sum_i x_i <= B, x_i >= 0

### 7.2 博弈论解概念

- **纳什均衡**: 每个智能体在给定其他智能体策略时最优
- **相关均衡**: 比纳什均衡更一般，允许协调
- **Shapley 值**: 合作博弈的公平分配方案
- **核 (Core)**: 联盟博弈中的稳定分配集合

### 7.3 信息论度量

- **互信息**: I(X;Y) = H(X) - H(X|Y)
- **信息增益**: 用于评估智能体动作的信息价值
- **熵正则化**: 鼓励探索的预算分配

## 参考实现资源

1. **EPyMARL**: https://github.com/uoe-agents/epymarl
   - 包含 VDN, QMIX, QPLEX, COMA 等实现
   
2. **PyMARL2**: https://github.com/hijkzzz/pymarl2
   - 改进的 MARL 实现框架
   
3. **SMAC**: https://github.com/oxwhirl/smac
   - 标准 MARL 基准测试环境
