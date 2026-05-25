# 轻量级多智能体协同训练系统

基于 MrlX 原理的简化复刻，使用 Qwen2.5-0.5B + LoRA，单卡 8GB 可跑。

## 架构

```
Main Agent (任务分解)     Sub Agent (子任务执行)
    |                              |
    v                              v
  LoRA 训练                     LoRA 训练
    |                              |
    +--------------+  +------------+
                   |  |
                   v  v
              数学任务环境
                   |
                   v
            双边奖励计算
                   |
                   v
            GRPO 训练循环
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `cotrain_system_main_sub.py` | 主训练程序（Main → Sub 任务分解模式） |
| `math_environment.py` | 数学任务环境（题目生成、答案验证、奖励计算） |
| `README.md` | 本文档 |

## 快速开始

```bash
cd lightweight_cotrain

# 安装依赖
pip install torch transformers peft accelerate

# 运行训练
python cotrain_system_main_sub.py
```

首次运行需下载 Qwen2.5-0.5B-Instruct 模型（约 1GB），脚本会自动从 ModelScope 下载。

## 配置参数

在 `cotrain_system_main_sub.py` 末尾修改：

```python
config = CoTrainConfig(
    main_model="./models/qwen/Qwen2___5-0___5B-Instruct",
    sub_model="./models/qwen/Qwen2___5-0___5B-Instruct",
    lora_r=8,                # LoRA rank
    lora_alpha=16,           # LoRA alpha
    lr=5e-4,                 # 学习率
    batch_size=2,            # 训练 batch size
    group_size=2,            # GRPO group 大小
    max_subtasks=2,          # 最多分解几个子任务
    max_response_len=256,    # 最大生成长度
    rollout_interval=1,      # 每几轮收集一次 rollout
    sync_interval=5,         # 每几轮同步保存权重
    save_dir="./cotrain_checkpoints_math",
    use_4bit=False,          # 是否启用 4bit 量化
    device="cuda:0",
)
```

## 训练流程

1. **Rollout 阶段**：加载两个 Agent，在数学环境上收集交互数据
2. **训练阶段**：卸载 Agent → 重新加载 → 执行 GRPO 策略更新
3. **同步阶段**：每 `sync_interval` 轮保存 LoRA 权重

## 硬件需求

| 配置 | 显存占用 | 说明 |
|------|---------|------|
| 单卡 8GB | ~2GB | 两个 0.5B + LoRA，时间片切换 |

## 与原版 MrlX 的对比

| 特性 | 原版 MrlX | 本轻量版 |
|------|----------|---------|
| 训练框架 | Megatron-LM + slime | Transformers + 自定义循环 |
| 推理服务 | SGLang | Transformers generate |
| Agent 关系 | 多角色协作 | Main → Sub 任务分解 |
| 分布式 | Ray + 多容器 | 本机单进程 |
| 最小模型 | 8B | 0.5B |
| 最小 GPU | 8× H20 | 单卡 8GB |
| 外部 API | 多个必须 | 无（自包含奖励函数） |
