# Algorithm 模組

SARSA(λ) + 線性函數近似，用於掃雷 RL 作業。

## 檔案

```
algorithm/
├── features.py   # 特徵萃取：棋盤 → φ(s, a)
├── agent.py      # SARSALambdaAgent
└── test_agent.py # 煙霧測試
```

## 對外介面

```python
from algorithm.agent import SARSALambdaAgent

agent = SARSALambdaAgent(alpha=0.01, gamma=0.99, lam=0.8, epsilon=0.3)

# 每局開始
agent.reset_trace()

# 每步
action = agent.select_action(state, valid_actions)   # 回傳 (r, c)
agent.update(s, a, reward, s_next, a_next, done)     # done=True 時後兩個傳 None

# ε 衰減（訓練迴圈負責呼叫）
agent.epsilon = max(0.05, agent.epsilon * 0.995)
```

## 特徵（7 維）

| 特徵 | 說明 |
|------|------|
| f1 | 周圍未翻開格數 / 8 |
| f2 | 周圍數字格總和 / 64 |
| f3 | 危險度 = 數字總和 / 未翻開格數 / 8 |
| f4 | 完全孤立？（0 或 1） |
| f5 | 全局未翻開格佔比 |
| f6 | 全局已知地雷佔比 |
| f7 | Bias = 1.0 |

## 超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| alpha | 0.01 | 學習率 |
| gamma | 0.99 | 折扣因子 |
| lam | 0.8 | trace 衰減速度 |
| epsilon | 0.3 | 初始探索率，訓練時逐漸衰減到 0.05 |