"""
Tkinter GUI for human play and agent replay.
"""

from __future__ import annotations

import time
import tkinter as tk

from algorithm.agent import SARSALambdaAgent
from environment.minesweeper_env import MinesweeperEnv


PANEL_BG = "#c0c0c0"
HIDDEN_BG = "#c0c0c0"
REVEALED_BG = "#d9d9d9"
MINE_BG = "#ff6b6b"
MINE_REVEAL_BG = "#f3c9c9"
HIGHLIGHT_BG = "#b7dfff"
FLAG_FG = "#c21807"
LIGHT_EDGE = "#ffffff"
DARK_EDGE = "#7b7b7b"
BOARD_BG = "#808080"
NUMBER_COLORS = {
    1: "#0000fe",
    2: "#017f01",
    3: "#fe0000",
    4: "#010080",
    5: "#800000",
    6: "#008080",
    7: "#000000",
    8: "#808080",
}


def _create_root(title: str) -> tk.Tk:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("无法启动 GUI，请确认当前环境支持桌面窗口，或改用 --ui cli。") from exc
    root.title(title)
    root.configure(bg=PANEL_BG)
    root.resizable(False, False)
    return root


class _BaseMinesweeperGUI:
    def __init__(self, root: tk.Tk, env: MinesweeperEnv, title: str):
        self.root = root
        self.env = env
        self.root.title(title)

        self._flags: set[tuple[int, int]] = set()
        self._board_canvas: tk.Canvas | None = None
        self._clock_after_id: str | None = None
        self._closed = False
        self._start_time: float | None = None
        self._elapsed_seconds = 0
        self._reveal_all_mines = False
        self._last_action: tuple[int, int] | None = None
        self.cell_size = self._choose_cell_size()
        self.board_width = self.env.cols * self.cell_size
        self.board_height = self.env.rows * self.cell_size
        self.info_width = max(360, self.board_width)

        self.mine_var = tk.StringVar(value=self._format_counter(self.env.num_mines))
        self.time_var = tk.StringVar(value="000")
        self.face_var = tk.StringVar(value=":)")
        self.status_var = tk.StringVar(value="")
        self.detail_var = tk.StringVar(value="")

        self._build_layout()
        self.root.update_idletasks()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _choose_cell_size(self) -> int:
        max_side = max(self.env.rows, self.env.cols)
        if max_side <= 9:
            return 32
        if max_side <= 16:
            return 28
        return 24

    def _build_layout(self):
        outer = tk.Frame(self.root, bg=PANEL_BG, bd=4, relief=tk.RAISED)
        outer.pack(padx=12, pady=12)

        header = tk.Frame(outer, bg=PANEL_BG, bd=3, relief=tk.SUNKEN)
        header.pack(fill="x", pady=(0, 8))

        header_controls = tk.Frame(header, bg=PANEL_BG)
        header_controls.pack(pady=8)

        mine_label = tk.Label(
            header_controls,
            textvariable=self.mine_var,
            width=5,
            font=("Consolas", 16, "bold"),
            fg="red",
            bg="black",
            bd=2,
            relief=tk.SUNKEN,
        )
        mine_label.grid(row=0, column=0, padx=8)

        self.reset_button = tk.Button(
            header_controls,
            textvariable=self.face_var,
            width=4,
            font=("Consolas", 14, "bold"),
            bg=HIDDEN_BG,
            relief=tk.RAISED,
            bd=3,
            takefocus=False,
        )
        self.reset_button.grid(row=0, column=1, padx=10)

        time_label = tk.Label(
            header_controls,
            textvariable=self.time_var,
            width=5,
            font=("Consolas", 16, "bold"),
            fg="red",
            bg="black",
            bd=2,
            relief=tk.SUNKEN,
        )
        time_label.grid(row=0, column=2, padx=8)

        info_frame = tk.Frame(
            outer,
            bg=PANEL_BG,
            width=self.info_width,
            height=64,
        )
        info_frame.pack(fill="x", pady=(0, 8))
        info_frame.pack_propagate(False)

        status_label = tk.Label(
            info_frame,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            bg=PANEL_BG,
            font=("Segoe UI", 10, "bold"),
            wraplength=self.info_width,
        )
        status_label.pack(fill="x")

        detail_label = tk.Label(
            info_frame,
            textvariable=self.detail_var,
            anchor="w",
            justify="left",
            bg=PANEL_BG,
            font=("Segoe UI", 9),
            wraplength=self.info_width,
        )
        detail_label.pack(fill="x", pady=(2, 0))

        board_border = tk.Frame(outer, bg=PANEL_BG, bd=3, relief=tk.SUNKEN)
        board_border.pack()

        self._board_canvas = tk.Canvas(
            board_border,
            width=self.board_width,
            height=self.board_height,
            bg=BOARD_BG,
            highlightthickness=0,
            bd=0,
        )
        self._board_canvas.pack()
        self._board_canvas.bind("<Button-1>", self._handle_canvas_left_click)
        self._board_canvas.bind("<Button-3>", self._handle_canvas_right_click)

    def _format_counter(self, value: int) -> str:
        if value >= 0:
            return f"{min(value, 999):03d}"
        return f"-{min(abs(value), 99):02d}"

    def _reset_clock(self):
        if self._clock_after_id is not None:
            self.root.after_cancel(self._clock_after_id)
            self._clock_after_id = None
        self._start_time = None
        self._elapsed_seconds = 0
        self.time_var.set("000")

    def _start_clock(self):
        if self._start_time is not None:
            return
        self._start_time = time.perf_counter()
        self._tick_clock()

    def _tick_clock(self):
        if self._closed or self._start_time is None:
            return
        self._elapsed_seconds = min(999, int(time.perf_counter() - self._start_time))
        self.time_var.set(f"{self._elapsed_seconds:03d}")
        self._clock_after_id = self.root.after(200, self._tick_clock)

    def _stop_clock(self):
        if self._clock_after_id is not None:
            self.root.after_cancel(self._clock_after_id)
            self._clock_after_id = None
        if self._start_time is not None:
            self._elapsed_seconds = min(999, int(time.perf_counter() - self._start_time))
            self.time_var.set(f"{self._elapsed_seconds:03d}")
            self._start_time = None

    def _prepare_board(self):
        self._flags.clear()
        self._reveal_all_mines = False
        self._last_action = None
        self.face_var.set(":)")
        self._reset_clock()
        self.mine_var.set(self._format_counter(self.env.num_mines))

    def _update_mine_counter(self):
        self.mine_var.set(self._format_counter(self.env.num_mines - len(self._flags)))

    def _cell_from_event(self, event) -> tuple[int, int] | None:
        row = event.y // self.cell_size
        col = event.x // self.cell_size
        if not self.env._is_valid(row, col):
            return None
        return row, col

    def _handle_canvas_left_click(self, event):
        cell = self._cell_from_event(event)
        if cell is None:
            return
        self.on_left_click(*cell)

    def _handle_canvas_right_click(self, event):
        cell = self._cell_from_event(event)
        if cell is None:
            return
        self.on_right_click(*cell)

    def _draw_hidden_cell(self, canvas: tk.Canvas, x0: int, y0: int, x1: int, y1: int, fill: str):
        canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=DARK_EDGE, width=1)
        canvas.create_line(x0, y0, x1 - 1, y0, fill=LIGHT_EDGE, width=2)
        canvas.create_line(x0, y0, x0, y1 - 1, fill=LIGHT_EDGE, width=2)
        canvas.create_line(x0 + 1, y1 - 1, x1, y1 - 1, fill=DARK_EDGE, width=2)
        canvas.create_line(x1 - 1, y0 + 1, x1 - 1, y1, fill=DARK_EDGE, width=2)

    def _draw_revealed_cell(
        self,
        canvas: tk.Canvas,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        fill: str,
    ):
        canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=DARK_EDGE, width=1)

    def _render_board(self):
        if self._board_canvas is None:
            return

        canvas = self._board_canvas
        canvas.delete("all")
        font_size = max(12, int(self.cell_size * 0.45))

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                is_visible = self.env._visible[r][c]
                is_mine = bool(self.env._mine_map[r][c])
                number = self.env._number_map[r][c]
                x0 = c * self.cell_size
                y0 = r * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size

                text = ""
                fg = "black"
                bg = HIDDEN_BG
                hidden_style = True

                if is_visible:
                    bg = REVEALED_BG
                    hidden_style = False
                    if is_mine:
                        text = "*"
                        bg = MINE_BG if self.env._done and not self.env._win else REVEALED_BG
                    elif number > 0:
                        text = str(number)
                        fg = NUMBER_COLORS.get(number, "black")
                elif self._reveal_all_mines and is_mine:
                    text = "*"
                    bg = MINE_REVEAL_BG
                    hidden_style = False
                elif self._reveal_all_mines and (r, c) in self._flags and not is_mine:
                    text = "X"
                    fg = FLAG_FG
                elif (r, c) in self._flags:
                    text = "F"
                    fg = FLAG_FG

                if self._last_action == (r, c) and not (self.env._done and is_mine):
                    bg = HIGHLIGHT_BG

                if hidden_style:
                    self._draw_hidden_cell(canvas, x0, y0, x1, y1, bg)
                else:
                    self._draw_revealed_cell(canvas, x0, y0, x1, y1, bg)

                if text:
                    canvas.create_text(
                        x0 + self.cell_size / 2,
                        y0 + self.cell_size / 2,
                        text=text,
                        fill=fg,
                        font=("Consolas", font_size, "bold"),
                    )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._reset_clock()
        self.root.destroy()

    def on_left_click(self, row: int, col: int):
        raise NotImplementedError

    def on_right_click(self, row: int, col: int):
        return None


class HumanMinesweeperGUI(_BaseMinesweeperGUI):
    def __init__(self, root: tk.Tk, env: MinesweeperEnv):
        title = f"Minesweeper {env.rows}x{env.cols} / {env.num_mines} mines"
        super().__init__(root, env, title)
        self.session_wins = 0
        self.session_losses = 0
        self.reset_button.configure(command=self.start_new_game)
        self.start_new_game()

    def _update_session_detail(self):
        total = self.session_wins + self.session_losses
        if total == 0:
            self.detail_var.set("战绩：0 胜 0 负")
            return
        win_rate = self.session_wins / total * 100
        self.detail_var.set(
            f"战绩：{self.session_wins} 胜 {self.session_losses} 负，胜率 {win_rate:.1f}%"
        )

    def start_new_game(self):
        self.env.reset()
        self._prepare_board()
        self.status_var.set(
            f"左键翻开，右键插旗。棋盘 {self.env.rows}x{self.env.cols}，地雷 {self.env.num_mines}。"
        )
        self._update_session_detail()
        self._render_board()

    def on_left_click(self, row: int, col: int):
        if self.env._done or (row, col) in self._flags or self.env._visible[row][col]:
            return

        self._start_clock()
        self.face_var.set(":|")
        self._last_action = (row, col)
        state, _reward, done, info = self.env.step((row, col))

        if done:
            self._stop_clock()
            if info.get("win"):
                self.session_wins += 1
                self.face_var.set("8)")
                self.status_var.set(
                    f"你赢了。用时 {self._elapsed_seconds}s，翻开 {state['trial_count']} 格。"
                )
            else:
                self.session_losses += 1
                self._reveal_all_mines = True
                self.face_var.set(":(")
                self.status_var.set(
                    f"踩雷，游戏结束。用时 {self._elapsed_seconds}s，翻开 {state['trial_count']} 格。"
                )
            self._update_session_detail()
        else:
            total_safe = self.env.rows * self.env.cols - self.env.num_mines
            opened = state["trial_count"]
            pct = opened / total_safe * 100 if total_safe > 0 else 0.0
            self.status_var.set(f"当前进度：已翻开 {opened}/{total_safe} 格（{pct:.0f}%）。")

        self._render_board()

    def on_right_click(self, row: int, col: int):
        if self.env._done or self.env._visible[row][col]:
            return "break"
        if (row, col) in self._flags:
            self._flags.remove((row, col))
        else:
            self._flags.add((row, col))
        self._update_mine_counter()
        self._render_board()
        return "break"


class ReplayMinesweeperGUI(_BaseMinesweeperGUI):
    def __init__(
        self,
        root: tk.Tk,
        env: MinesweeperEnv,
        agent: SARSALambdaAgent,
        episodes: int,
        delay: float,
        max_steps: int,
    ):
        title = f"Minesweeper Replay {env.rows}x{env.cols} / {env.num_mines} mines"
        super().__init__(root, env, title)
        self.agent = agent
        self.episodes = episodes
        self.delay_ms = max(1, int(delay * 1000))
        self.max_steps = max_steps
        self.current_episode = 0
        self.wins = 0
        self.step_idx = 0
        self.total_reward = 0.0
        self.state: dict | None = None
        self._replay_after_id: str | None = None

        self.reset_button.configure(command=self.restart_replay)
        self.restart_replay()

    def _cancel_replay(self):
        if self._replay_after_id is not None:
            self.root.after_cancel(self._replay_after_id)
            self._replay_after_id = None

    def _set_detail(self, action: tuple[int, int] | None = None):
        action_text = "-" if action is None else f"({action[0]}, {action[1]})"
        self.detail_var.set(
            f"第 {self.current_episode}/{self.episodes} 局 | 胜场 {self.wins} | "
            f"步数 {self.step_idx} | 动作 {action_text} | 总奖励 {self.total_reward:.3f}"
        )

    def restart_replay(self):
        self._cancel_replay()
        self.current_episode = 0
        self.wins = 0
        self.step_idx = 0
        self.total_reward = 0.0
        self.state = None
        self.env.reset()
        self._prepare_board()
        self.status_var.set("准备开始自动回放。点击中间按钮可重新 replay。")
        self._set_detail()
        self._render_board()
        self._replay_after_id = self.root.after(250, self._start_next_episode)

    def _start_next_episode(self):
        self._cancel_replay()
        if self.current_episode >= self.episodes:
            win_rate = self.wins / self.episodes * 100 if self.episodes > 0 else 0.0
            self.face_var.set("8)" if self.wins == self.episodes else ":)")
            self.status_var.set(f"回放完成：{self.wins}/{self.episodes} 胜（{win_rate:.1f}%）。")
            self._set_detail()
            return

        self.current_episode += 1
        self.step_idx = 0
        self.total_reward = 0.0
        self.state = self.env.reset()
        self._prepare_board()
        self._start_clock()
        self.status_var.set(f"正在回放第 {self.current_episode}/{self.episodes} 局。")
        self._set_detail()
        self._render_board()
        self._replay_after_id = self.root.after(self.delay_ms, self._advance_step)

    def _finish_episode(self, force_stop: bool = False):
        self._cancel_replay()
        self._stop_clock()
        win = bool(self.state and self.state.get("win"))
        if win:
            self.wins += 1
            self.face_var.set("8)")
            self.status_var.set(
                f"第 {self.current_episode} 局获胜，用时 {self._elapsed_seconds}s。"
            )
        elif force_stop:
            self.face_var.set(":|")
            self._reveal_all_mines = True
            self.status_var.set(
                f"第 {self.current_episode} 局达到最大步数 {self.max_steps}，提前结束。"
            )
        else:
            self.face_var.set(":(")
            self._reveal_all_mines = True
            self.status_var.set(
                f"第 {self.current_episode} 局失败，用时 {self._elapsed_seconds}s。"
            )
        self._set_detail(self._last_action)
        self._render_board()
        pause_ms = max(800, self.delay_ms * 3)
        self._replay_after_id = self.root.after(pause_ms, self._start_next_episode)

    def _advance_step(self):
        self._cancel_replay()
        if self.state is None:
            return
        if self.step_idx >= self.max_steps:
            self._finish_episode(force_stop=True)
            return

        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            self._finish_episode(force_stop=True)
            return

        action = self.agent.select_action(self.state, valid_actions)
        self._last_action = action
        self.state, reward, done, _info = self.env.step(action)
        self.step_idx += 1
        self.total_reward += reward
        self._set_detail(action)
        self._render_board()

        if done:
            self._finish_episode(force_stop=False)
            return

        self._replay_after_id = self.root.after(self.delay_ms, self._advance_step)

    def on_left_click(self, row: int, col: int):
        return None

    def close(self):
        self._cancel_replay()
        super().close()


def launch_human_gui(env: MinesweeperEnv):
    root = _create_root(f"Minesweeper {env.rows}x{env.cols}")
    HumanMinesweeperGUI(root, env)
    root.mainloop()


def launch_replay_gui(
    env: MinesweeperEnv,
    agent: SARSALambdaAgent,
    episodes: int,
    delay: float,
    max_steps: int,
):
    root = _create_root(f"Minesweeper Replay {env.rows}x{env.cols}")
    ReplayMinesweeperGUI(
        root=root,
        env=env,
        agent=agent,
        episodes=episodes,
        delay=delay,
        max_steps=max_steps,
    )
    root.mainloop()
