# Minesweeper RL — SARSA(λ) 扫雷强化学习

基于 SARSA(λ) + 线性函数近似的扫雷游戏强化学习项目，支持人工游玩、模型训练、评估与可视化回放。

---

## 项目结构

```
.
├── environment/              # 游戏环境模块
│   ├── minesweeper_env.py    # 扫雷环境（reset/step/get_valid_actions）
│   ├── gui.py                # Tkinter GUI 界面
│   └── test_env.py           # 环境单元测试
│
├── algorithm/                # 算法模块
│   ├── agent.py              # SARSALambdaAgent（SARSA(λ) + 线性近似）
│   ├── features.py           # 特征提取（10 维，含推理特征）
│   └── test_agent.py         # Agent 冒烟测试
│
├── training_evaluation/      # 训练与评估模块
│   ├── train.py              # 主训练脚本
│   ├── evaluate.py           # 模型评估脚本
│   ├── replay.py             # 回放脚本
│   ├── plot_training.py      # 训练曲线绘图
│   ├── plot_compare.py       # 对比图绘制工具
│   ├── compare_algorithms.py # 算法对比实验
│   ├── compare_baseline.py   # 与随机基线对比
│   ├── compare_difficulty.py # 不同难度对比
│   ├── compare_epsilon.py    # ε 超参消融实验
│   └── compare_lambda.py     # λ 超参消融实验
│
├── play.py                   # 人工游玩入口
└── requirements.txt          # Python 依赖
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

> 依赖：`numpy >= 1.20`，`matplotlib >= 3.7`（Python 3.10+）

### 人工游玩

```bash
# GUI 模式（默认）
python play.py

# CLI 模式
python play.py --ui cli
```

### 训练模型

```bash
# 默认配置（9×9，10 颗雷，20000 局）
python -m training_evaluation.train

# 自定义参数
python -m training_evaluation.train --episodes 30000 --rows 9 --cols 9 --mines 10 --lam 0.9 --tag myrun
```

训练产物保存在 `training_evaluation/runs/<timestamp>/`：

| 文件 | 内容 |
|------|------|
| `config.json` | 本次训练配置 |
| `log.csv` | 逐局记录（胜负、奖励、步数等） |
| `final_model.npz` | 训练完成的权重 |
| `training_summary.png` | 训练过程曲线图 |
| `training_report.md` | 胜率、奖励等汇总报告 |

### 评估模型

```bash
python -m training_evaluation.evaluate --run training_evaluation/runs/best
```

### 回放（观看 AI 对局）

```bash
# GUI 回放（默认）
python -m training_evaluation.replay --run training_evaluation/runs/best

# CLI 回放
python -m training_evaluation.replay --run training_evaluation/runs/best --ui cli
```

---

## 算法说明

### SARSA(λ)

采用 **on-policy SARSA(λ)**（资格迹）结合 **线性函数近似** 估计动作价值函数：

```
Q(s, a) = w · φ(s, a)
```

每步更新（Eligibility Trace）：

```
δ  = r + γ · Q(s', a') − Q(s, a)
e  ← γλe + φ(s, a)
w  ← w + α · δ · e
```

终局（done=True）时 `s'`、`a'` 传 `None`，`δ = r − Q(s, a)`。

### 特征向量（10 维）

| # | 特征 | 说明 |
|---|------|------|
| f1 | 周围隐藏格比例 | 未翻开邻居数 / 8 |
| f2 | 周围数字总和 | 邻居数字之和 / 64 |
| f3 | 危险度估计 | 数字总和 / 隐藏格数 / 8 |
| f4 | 完全孤立 | 周围无数字提示时为 1 |
| f5 | 全局隐藏格比例 | 剩余未翻开格 / 总格数 |
| f6 | 全局已知地雷比例 | 已爆炸地雷数 / 总格数 |
| f7 | Bias | 固定为 1.0 |
| f8 | 推理安全 | 逻辑推断该格安全时为 1 |
| f9 | 推理地雷 | 逻辑推断该格为雷时为 1 |
| f10 | 连续危险度 | 邻居数字格的剩余雷概率最大值 |

### 默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | 0.01 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.8 | Trace 衰减速度 λ |
| `eps_start` | 0.3 | 初始探索率 ε |
| `eps_end` | 0.05 | 最终探索率 |
| `eps_decay` | 0.9995 | 每局衰减系数 |

---

## 环境接口

```python
from environment import MinesweeperEnv

env = MinesweeperEnv(grid_size=(9, 9), num_mines=10)

state = env.reset()              # 重置，返回状态字典
valid = env.get_valid_actions()  # 返回所有未翻开格坐标列表
state, reward, done, info = env.step((r, c))  # 执行动作
```

状态字典 `state` 包含：

| 键 | 类型 | 说明 |
|----|------|------|
| `grid_size` | `(rows, cols)` | 棋盘尺寸 |
| `map` | `list[list[int]]` | 可见棋盘（-1=隐藏，0-8=数字，9=地雷） |
| `done` | `bool` | 是否结束 |
| `win` | `bool` | 是否胜利 |
| `reward` | `float` | 本步奖励 |
| `trial_count` | `int` | 已翻开格子总数 |

奖励设计：

| 事件 | 奖励 |
|------|------|
| 翻开安全格（每格） | +0.05 |
| 胜利 | +10.0 |
| 踩雷 | −10.0 |
| 重复翻开已知格 | −0.5 |

---

## 实验对比脚本

```bash
# 与随机基线对比
python -m training_evaluation.compare_baseline

# 不同难度（简单/中等/困难）对比
python -m training_evaluation.compare_difficulty

# λ 超参消融（λ = 0, 0.5, 0.8, 0.9）
python -m training_evaluation.compare_lambda

# ε 衰减策略消融
python -m training_evaluation.compare_epsilon

# 算法横向对比
python -m training_evaluation.compare_algorithms
```

---

## 分工

| 模块 | 负责内容 |
|------|----------|
| `environment/` | 游戏建模：扫雷环境、GUI、CLI 交互 |
| `algorithm/` | 算法实现：SARSA(λ) Agent、特征提取 |
| `training_evaluation/` | 训练流程、评估指标、可视化与消融实验 |
