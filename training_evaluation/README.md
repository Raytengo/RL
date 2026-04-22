# 模块三：训练与评估（Training & Evaluation）

## 文件一览

```
training_evaluation/
├── train.py              # 主训练脚本
├── evaluate.py           # 模型评估（贪婪策略）
├── replay.py             # 回放已训练模型的对局过程
├── plot_training.py      # 单次训练曲线绘图
├── plot_compare.py       # 对比实验绘图（通用）
├── compare_baseline.py   # 随机基线对比
├── compare_lambda.py     # λ 超参消融（λ ∈ {0.0, 0.4, 0.8, 1.0}）
├── compare_epsilon.py    # ε 策略消融（Fixed/Decay/Adaptive/Decay+Adaptive）
├── compare_algorithms.py # 算法对比（SARSA / Q-Learning / MonteCarlo）
└── compare_difficulty.py # 难度对比（4×4 / 6×6 / 9×9）
```

---

## 快速上手

### 训练

```bash
python -m training_evaluation.train                     # 默认配置
python -m training_evaluation.train --episodes 30000   # 自定义局数
python -m training_evaluation.train --rows 9 --cols 9 --mines 10 --lam 0.9 --tag exp1
```

产物保存在 `training_evaluation/runs/<timestamp>/`：

| 文件 | 说明 |
|------|------|
| `config.json` | 本次超参配置 |
| `log.csv` | 逐局记录（episode, win, reward, steps, epsilon, w_norm）|
| `final_model.npz` | 权重 `w`（10 维）|
| `training_summary.png` | 训练曲线图（权重柱状图 + reward + steps）|
| `training_report.md` | 胜率/奖励/步数汇总 |

### 评估

```bash
python -m training_evaluation.evaluate                             # 自动找最新 run
python -m training_evaluation.evaluate --run training_evaluation/runs/best --episodes 1000
```

### 回放

```bash
python -m training_evaluation.replay --run training_evaluation/runs/best           # GUI
python -m training_evaluation.replay --run training_evaluation/runs/best --ui cli  # CLI
```

---

## 对比实验

所有对比脚本产物统一保存在 `training_evaluation/eval_results/` 下。

### 随机基线

```bash
python -m training_evaluation.compare_baseline --test-episodes 1000 --seeds 42 43 44
```

### λ 消融

对比 λ ∈ {0.0, 0.4, 0.8, 1.0}：

```bash
python -m training_evaluation.compare_lambda
```

- λ=0：等价于单步 SARSA（one-step TD）
- λ=0.8：项目默认值
- λ=1.0：接近 Monte Carlo 行为

### ε 策略消融（2×2 设计）

```
                 | 跨局不衰减  | 跨局衰减 (0.3→0.05)
  ----------------|------------|--------------------
  局内不自适应    |  Fixed     |  Decay
  局内自适应(B)   |  Adaptive  |  Decay+Adaptive
```

```bash
python -m training_evaluation.compare_epsilon
```

局内自适应公式（方案 B）：

```
eps_local = eps_base × (1 − opened_ratio)²
```

### 算法横向对比

```bash
python -m training_evaluation.compare_algorithms               # 默认 Decay 策略
python -m training_evaluation.compare_algorithms --eps-strategy decay_adaptive
```

| 算法 | 特征 |
|------|------|
| SARSA(λ) | on-policy，bootstrapping，资格迹 λ=0.8 |
| Q-Learning | off-policy，bootstrapping，无 trace |
| MonteCarlo | on-policy，不 bootstrapping，整局后更新 |

### 难度对比

```bash
python -m training_evaluation.compare_difficulty               # 4×4 / 6×6 / 9×9
python -m training_evaluation.compare_difficulty --episodes 10000  # 统一局数
```

---

## 可视化

```bash
# 单次训练曲线
python -m training_evaluation.plot_training --run training_evaluation/runs/best

# 对比实验图（4 张：训练曲线/胜率柱/accuracy-time/多指标）
python -m training_evaluation.plot_compare --run training_evaluation/eval_results/eps_compare_<timestamp>
```

---

## 快速 smoke test

```bash
# 所有对比实验，只跑 500 局 × 1 seed，约 5 分钟验证管道可用
python -m training_evaluation.compare_lambda     --episodes 500 --test-episodes 200 --seeds 42
python -m training_evaluation.compare_epsilon    --episodes 500 --test-episodes 200 --seeds 42
python -m training_evaluation.compare_algorithms --episodes 500 --test-episodes 200 --seeds 42
python -m training_evaluation.compare_difficulty --episodes 500
```

或者直接用根目录的一键脚本：

```bash
python run_all.py --quick
```
