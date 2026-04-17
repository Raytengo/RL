"""
扫雷游戏环境
提供两种模式：
  - 人工 CLI 游玩：调用 play_cli()
  - 机器训练接口：使用 reset() / step() / get_valid_actions()
"""

import random
import copy


# 棋盘格子状态常量（map 数组中的值）
HIDDEN = -1   # 未翻开
MINE = 9      # 地雷（爆炸后可见）

# 默认 reward shaping。保持小的过程奖励，但让终局胜负占主导。
DEFAULT_SAFE_OPEN_REWARD_PER_CELL = 0.05
DEFAULT_WIN_REWARD = 10.0
DEFAULT_LOSE_REWARD = -10.0
DEFAULT_REPEAT_REWARD = -0.5


class MinesweeperEnv:
    def __init__(
        self,
        grid_size: tuple = (9, 9),
        num_mines: int = 10,
        safe_open_reward_per_cell: float = DEFAULT_SAFE_OPEN_REWARD_PER_CELL,
        win_reward: float = DEFAULT_WIN_REWARD,
        lose_reward: float = DEFAULT_LOSE_REWARD,
        repeat_reward: float = DEFAULT_REPEAT_REWARD,
    ):
        """
        grid_size: (rows, cols)
        num_mines: 地雷数量
        """
        self.rows, self.cols = grid_size
        self.grid_size = grid_size
        self.num_mines = num_mines
        self.safe_open_reward_per_cell = safe_open_reward_per_cell
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.repeat_reward = repeat_reward

        # 内部状态（由 reset() 初始化）
        self._mine_map = None      # 真实地雷分布，True=地雷
        self._number_map = None    # 每格周围地雷数（0-8），地雷格为-1
        self._visible = None       # 玩家可见状态，True=已翻开
        self._trial_count = 0
        self._done = False
        self._win = False

        self.reset()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """重置游戏，返回初始状态字典。"""
        self._mine_map = [[False] * self.cols for _ in range(self.rows)]
        self._number_map = [[0] * self.cols for _ in range(self.rows)]
        self._visible = [[False] * self.cols for _ in range(self.rows)]
        self._trial_count = 0
        self._done = False
        self._win = False

        self._place_mines()
        self._compute_numbers()

        return self._get_state()

    def step(self, action: tuple) -> tuple:
        """
        执行一步动作（翻开某个格子）。
        action: (row, col)
        返回: (state, reward, done, info)
        """
        if self._done:
            raise RuntimeError("游戏已结束，请先调用 reset()。")

        row, col = action
        if not self._is_valid(row, col):
            raise ValueError(f"坐标越界：{action}")
        if self._visible[row][col]:
            # 翻已翻开的格子，给小惩罚
            reward = self.repeat_reward
            state = self._get_state(reward=reward)
            return state, reward, False, {"win": False, "repeated": True}

        if self._mine_map[row][col]:
            # 踩雷
            self._visible[row][col] = True
            self._trial_count += 1
            self._done = True
            self._win = False
            reward = self.lose_reward
        else:
            # 安全：递归展开
            newly_opened = self._reveal(row, col)
            self._trial_count += newly_opened
            reward = self.safe_open_reward_per_cell * newly_opened

            # 胜利判断：所有非地雷格子都已翻开
            if self._check_win():
                self._done = True
                self._win = True
                reward += self.win_reward

        state = self._get_state(reward=reward)
        info = {"win": self._win}
        return state, reward, self._done, info

    def get_valid_actions(self) -> list:
        """返回所有未翻开格子的坐标列表。"""
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if not self._visible[r][c]
        ]

    def render(self):
        """在终端打印当前棋盘（调试 / 人工模式用）。"""
        print(self._board_string())

    # ------------------------------------------------------------------
    # 人工 CLI 游玩
    # ------------------------------------------------------------------

    def play_cli(self):
        """交互式命令行扫雷。"""
        print("=== 扫雷 CLI 模式 ===")
        print(f"棋盘大小：{self.rows}x{self.cols}，地雷数：{self.num_mines}")
        print("输入格式：行 列（从 0 开始），输入 q 退出\n")

        self.reset()
        while not self._done:
            self.render()
            user_input = input("请输入坐标：").strip()
            if user_input.lower() == "q":
                print("已退出。")
                return
            try:
                r, c = map(int, user_input.split())
            except ValueError:
                print("格式错误，请重新输入（例如：3 4）")
                continue

            if not self._is_valid(r, c):
                print(f"坐标越界，请输入 0-{self.rows-1} 行，0-{self.cols-1} 列。")
                continue

            if self._visible[r][c]:
                print("该格子已翻开，请选择其他格子。")
                continue

            _, _, done, info = self.step((r, c))
            if done:
                self.render()
                if info["win"]:
                    print(f"恭喜！你赢了！共翻开 {self._trial_count} 个格子。")
                else:
                    print("很遗憾，踩雷了！游戏结束。")
                return

        self.render()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _place_mines(self):
        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        mine_cells = random.sample(all_cells, self.num_mines)
        for r, c in mine_cells:
            self._mine_map[r][c] = True

    def _compute_numbers(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self._mine_map[r][c]:
                    self._number_map[r][c] = -1
                    continue
                count = 0
                for dr, dc in self._neighbors(r, c):
                    if self._mine_map[dr][dc]:
                        count += 1
                self._number_map[r][c] = count

    def _reveal(self, row, col) -> int:
        """
        BFS 递归翻开格子，遇到数字格停止展开。
        返回本次新翻开的格子数。
        """
        if self._visible[row][col]:
            return 0

        queue = [(row, col)]
        self._visible[row][col] = True
        opened = 1

        while queue:
            r, c = queue.pop()
            if self._number_map[r][c] == 0:
                for nr, nc in self._neighbors(r, c):
                    if not self._visible[nr][nc] and not self._mine_map[nr][nc]:
                        self._visible[nr][nc] = True
                        opened += 1
                        queue.append((nr, nc))

        return opened

    def _check_win(self) -> bool:
        for r in range(self.rows):
            for c in range(self.cols):
                if not self._mine_map[r][c] and not self._visible[r][c]:
                    return False
        return True

    def _neighbors(self, r, c):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if self._is_valid(nr, nc):
                    yield nr, nc

    def _is_valid(self, r, c) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _get_state(self, reward: float = 0.0) -> dict:
        """构造并返回标准状态字典。"""
        visible_map = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if not self._visible[r][c]:
                    row.append(HIDDEN)
                elif self._mine_map[r][c]:
                    row.append(MINE)
                else:
                    row.append(self._number_map[r][c])
            visible_map.append(row)

        return {
            "grid_size": self.grid_size,
            "map": visible_map,
            "done": self._done,
            "win": self._win,
            "reward": reward,
            "trial_count": self._trial_count,
        }

    def _board_string(self) -> str:
        """生成棋盘的字符串表示。"""
        col_header = "    " + "  ".join(f"{c:2d}" for c in range(self.cols))
        lines = [col_header, "   +" + "---" * self.cols + "+"]
        for r in range(self.rows):
            row_str = f"{r:2d} |"
            for c in range(self.cols):
                if not self._visible[r][c]:
                    cell = " ."
                elif self._mine_map[r][c]:
                    cell = " *"
                elif self._number_map[r][c] == 0:
                    cell = "  "
                else:
                    cell = f" {self._number_map[r][c]}"
                row_str += cell + " "
            row_str += "|"
            lines.append(row_str)
        lines.append("   +" + "---" * self.cols + "+")
        return "\n".join(lines)
