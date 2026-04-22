# 模块二：算法实现（Algorithm）

SARSA(λ) + 线性函数近似，用于扫雷 RL 作业。

## 文件

```
algorithm/
├── features.py   # 特征提取：棋盘 → φ(s, a)，10 维
├── agent.py      # SARSALambdaAgent
└── test_agent.py # 冒烟测试
```

## 对外接口

```python
from algorithm.agent import SARSALambdaAgent

agent = SARSALambdaAgent(alpha=0.01, gamma=0.99, lam=0.8, epsilon=0.3)

# 每局开始
agent.reset_trace()

# 每步
action = agent.select_action(state, valid_actions)   # 返回 (r, c)
agent.update(s, a, reward, s_next, a_next, done)     # done=True 时后两个传 None

# ε 衰减（训练循环负责调用）
agent.epsilon = max(0.05, agent.epsilon * 0.9995)
```

## 特征向量（10 维）

| # | 特征 | 计算方式 |
|---|------|---------|
| f1 | 周围未翻开格比例 | 隐藏邻居数 / 8 |
| f2 | 周围数字总和 | 邻居数字之和 / 64 |
| f3 | 危险度估计 | 数字总和 / 隐藏格数 / 8 |
| f4 | 完全孤立 | 周围无数字提示时为 1 |
| f5 | 全局未翻开比例 | 剩余隐藏格 / 总格数 |
| f6 | 全局已知地雷比例 | 已爆炸地雷数 / 总格数 |
| f7 | Bias | 固定为 1.0 |
| f8 | **推理安全** | 逻辑推断该格安全时为 1 |
| f9 | **推理地雷** | 逻辑推断该格为雷时为 1 |
| f10 | **连续危险度** | 邻居数字格剩余雷概率最大值 |

> f8/f9/f10 使用单遍扫描推理（`_infer_mines` + `_infer_safe`）。同一 state 只推理一次（按棋盘 tuple 缓存），不重复计算。

## 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| alpha | 0.01 | 学习率 |
| gamma | 0.99 | 折扣因子 |
| lam | 0.8 | trace 衰减速度 |
| epsilon | 0.3 | 初始探索率，训练时逐渐衰减到 0.05 |

## 算法更新公式

```
δ  = r + γ·Q(s', a') − Q(s, a)   # TD 误差
e  ← γλe + φ(s, a)               # 资格迹更新
w  ← w + α·δ·e                   # 权重更新
```

终局（done=True）时：`δ = r − Q(s, a)`（无 bootstrap）
