# 模块一：游戏建模（Environment）

负责人：[你的名字]

## 任务

实现扫雷游戏的环境，作为 RL 训练的"世界模型"。

## 需要实现的内容

1. **游戏逻辑核心**
   - 地雷随机布置
   - 翻开格子（含递归展开空白区域）
   - 游戏胜负判断

2. **人工游玩模式（CLI）**
   - 用户通过命令行输入坐标进行扫雷
   - 打印当前棋盘状态

3. **机器训练模式（Environment API）**
   - 实现下方"对外接口"中规定的标准接口
   - 供算法模块和训练模块调用

## 对外接口（必须实现，供其他模块调用）

```python
class MinesweeperEnv:
    def __init__(self, grid_size: tuple, num_mines: int):
        """
        grid_size: (rows, cols)，例如 (9, 9)
        num_mines: 地雷总数，例如 10
        """
        ...

    def reset(self) -> dict:
        """
        重置游戏，返回初始状态。
        返回值格式见"状态字典格式"。
        """
        ...

    def step(self, action: tuple) -> tuple[dict, float, bool, dict]:
        """
        执行一步动作。
        action: (row, col)，表示要翻开的格子坐标
        返回: (state, reward, done, info)
          - state: 状态字典
          - reward: 本步奖励（float）
          - done: 是否结束（bool）
          - info: 附加信息字典，例如 {"win": True/False}
        """
        ...

    def get_valid_actions(self) -> list[tuple]:
        """
        返回当前所有合法动作（未翻开的格子坐标列表）。
        """
        ...

    def render(self):
        """
        在终端打印当前棋盘（人工模式和调试用）。
        """
        ...
```

## 状态字典格式

`reset()` 和 `step()` 返回的 state 统一为：

```python
{
    "grid_size": (rows, cols),          # 棋盘尺寸
    "map": [[int, ...], ...],           # 当前可见棋盘，-1=未知，0-8=数字，9=地雷（爆炸时）
    "done": bool,                       # 游戏是否结束
    "win": bool,                        # 是否获胜（done=False 时无意义）
    "reward": float,                    # 本步奖励
    "trial_count": int                  # 已翻开格子数
}
```

## 奖励设计（可自行调整）

奖励的具体数值由本模块决定，但建议：
- 踩雷 → 负奖励（游戏结束）
- 安全翻开 → 正奖励
- 获胜 → 大正奖励

## 依赖

只需 Python 标准库（`random`, `copy` 等），不需要第三方库。
