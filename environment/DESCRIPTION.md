# 模块一：游戏建模（Environment）

## 概述

本模块实现了扫雷游戏的完整环境，作为整个 RL 系统的"世界模型"。它向上层算法暴露标准的强化学习接口（`reset / step / get_valid_actions`），同时提供一个人工 CLI 游玩模式用于验证游戏逻辑的正确性。

---

## 文件结构

```
environment/
├── minesweeper_env.py   # 核心实现
├── __init__.py          # 包入口
├── play.py              # CLI 游玩入口脚本
├── test_env.py          # 测试套件（47 项）
├── DESCRIPTION.md       # 本文档
└── README.md            # 接口说明（供队友参考）
```

---

## 游戏规则

扫雷是一个信息不完全的单人决策游戏：

- 棋盘由 `rows × cols` 个格子组成，其中随机埋有若干地雷。
- 每步玩家选择一个**未翻开**的格子翻开：
  - 若该格是地雷，游戏**失败**结束。
  - 若该格安全，显示其周围 8 格中的地雷数量（0–8）。
  - 若该格数字为 0（周围无雷），**自动递归翻开**相邻所有安全格（BFS 连锁展开）。
- 当所有**非地雷格子**均被翻开时，游戏**胜利**结束。

---

## 实现细节

### 棋盘表示

内部维护三张二维数组：

| 数组 | 含义 |
|------|------|
| `_mine_map` | 真实地雷分布，`True` = 地雷 |
| `_number_map` | 每格周围地雷数（0–8），地雷格为 `-1` |
| `_visible` | 玩家可见状态，`True` = 已翻开 |

对外暴露的 `state["map"]` 使用统一编码：

| 值 | 含义 |
|----|------|
| `-1` | 未翻开（`HIDDEN`） |
| `0–8` | 已翻开，数字为周围地雷数 |
| `9` | 地雷（仅在踩雷后显示，`MINE`） |

### BFS 连锁展开

翻开一个数字为 `0` 的格子时，使用迭代 BFS 自动展开所有相连的安全格，直到遇到数字格为止（数字格本身会被翻开但不继续扩散）。这与标准扫雷行为一致。

### 奖励设计

| 情形 | 奖励 |
|------|------|
| 踩雷（游戏失败） | `-1.0` |
| 安全翻开 `k` 格 | `0.3 + 0.1 × (k - 1)` |
| 全部翻完（游戏胜利） | `+1.0` |
| 重复翻已开的格子 | `-0.5` |

安全步的奖励随连锁翻开数线性增长，鼓励算法优先选择能引发大面积展开的格子。

---

## 对外接口

```python
from environment import MinesweeperEnv

env = MinesweeperEnv(grid_size=(9, 9), num_mines=10)

state = env.reset()                        # 重置，返回初始状态字典
actions = env.get_valid_actions()          # 获取合法动作列表 [(r, c), ...]
state, reward, done, info = env.step((r, c))  # 执行一步
env.render()                               # 打印当前棋盘
```

返回的 `state` 字典格式：

```python
{
    "grid_size":   (rows, cols),   # 棋盘尺寸
    "map":         [[int]],        # 可见棋盘（-1/0-8/9）
    "done":        bool,           # 游戏是否结束
    "win":         bool,           # 是否胜利
    "reward":      float,          # 本步奖励（step() 填充）
    "trial_count": int             # 累计已翻开格子数
}
```

---

## 测试

运行全部 47 项测试：

```bash
python3 environment/test_env.py
```

覆盖范围：初始化、reset、step（安全/踩雷/重复/越界/越界异常）、胜利判定、BFS 展开、number_map 正确性、500 局随机压力测试、render 输出。

---

## CLI 游玩

```bash
# 默认 9×9，10 颗地雷
python3 play.py

# 自定义参数
python3 play.py --rows 16 --cols 16 --mines 40
```
