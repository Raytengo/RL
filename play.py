#!/usr/bin/env python3
"""
扫雷 CLI 游玩入口

用法：
  python3 play.py                          # 默认 9×9，10 颗地雷
  python3 play.py --rows 16 --cols 16 --mines 40
  python3 play.py --rows 9 --cols 9 --mines 10 --no-color
"""

import argparse
import sys
import time
from pathlib import Path

# 支持从项目根目录或 environment/ 目录下运行
sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment.minesweeper_env import MinesweeperEnv, HIDDEN, MINE

# ── ANSI 颜色 ──────────────────────────────────────────────
COLORS = {
    1: "\033[94m",   # 蓝
    2: "\033[92m",   # 绿
    3: "\033[91m",   # 红
    4: "\033[34m",   # 深蓝
    5: "\033[31m",   # 深红
    6: "\033[96m",   # 青
    7: "\033[35m",   # 紫
    8: "\033[90m",   # 灰
}
RESET   = "\033[0m"
BOLD    = "\033[1m"
RED_BG  = "\033[41m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
GRAY    = "\033[90m"


def render_pretty(env, use_color: bool):
    """彩色棋盘渲染。"""
    rows, cols = env.rows, env.cols
    # 列号标题
    header = "      " + "".join(f"{c:3d}" for c in range(cols))
    sep    = "    +" + "───" * cols + "+"
    lines  = [header, sep]

    for r in range(rows):
        row_str = f" {r:2d} │"
        for c in range(cols):
            cell_val = None
            if not env._visible[r][c]:
                ch = " ·"
                if use_color:
                    ch = GRAY + " ·" + RESET
            elif env._mine_map[r][c]:
                ch = " *"
                if use_color:
                    ch = RED_BG + BOLD + " *" + RESET
            elif env._number_map[r][c] == 0:
                ch = "  "
            else:
                n = env._number_map[r][c]
                ch = f" {n}"
                if use_color and n in COLORS:
                    ch = COLORS[n] + BOLD + f" {n}" + RESET
            row_str += ch + " "
        row_str += "│"
        lines.append(row_str)

    lines.append(sep)
    print("\n".join(lines))


def print_status(env, use_color: bool):
    """打印局面信息栏。"""
    total_safe = env.rows * env.cols - env.num_mines
    opened = env._trial_count
    remaining_mines = env.num_mines  # 玩家视角：雷数不变（标准扫雷无标旗）
    pct = opened / total_safe * 100 if total_safe > 0 else 0
    bar_len = 20
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    if use_color:
        bar = GREEN + "█" * filled + RESET + "░" * (bar_len - filled)
    print(f"  进度 [{bar}] {pct:.0f}%   "
          f"已开 {opened}/{total_safe}   "
          f"地雷 {remaining_mines} 颗")


def parse_args():
    p = argparse.ArgumentParser(description="扫雷 CLI 游玩")
    p.add_argument("--rows",     type=int, default=9,  help="棋盘行数（默认 9）")
    p.add_argument("--cols",     type=int, default=9,  help="棋盘列数（默认 9）")
    p.add_argument("--mines",    type=int, default=10, help="地雷数量（默认 10）")
    p.add_argument("--no-color", action="store_true",  help="禁用彩色输出")
    return p.parse_args()


def main():
    args = parse_args()
    use_color = not args.no_color

    # 验证参数
    total = args.rows * args.cols
    if args.mines >= total:
        print(f"错误：地雷数 ({args.mines}) 必须小于总格子数 ({total})。")
        sys.exit(1)
    if args.rows < 2 or args.cols < 2:
        print("错误：棋盘至少 2×2。")
        sys.exit(1)

    env = MinesweeperEnv(grid_size=(args.rows, args.cols), num_mines=args.mines)

    session_wins   = 0
    session_losses = 0

    while True:
        # ── 新局开始 ──
        env.reset()
        start_time = time.time()

        print("\n" + "═" * (args.cols * 3 + 8))
        if use_color:
            print(f"  {BOLD}💣  扫雷  {args.rows}×{args.cols}  地雷 {args.mines} 颗{RESET}")
        else:
            print(f"  扫雷  {args.rows}×{args.cols}  地雷 {args.mines} 颗")
        print(f"  输入坐标翻格（行 列，从 0 开始）  |  r = 重开  |  q = 退出")
        print("═" * (args.cols * 3 + 8))

        game_over = False
        while not game_over:
            print()
            render_pretty(env, use_color)
            print_status(env, use_color)
            print()

            try:
                raw = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                sys.exit(0)

            if raw.lower() == "q":
                print("已退出。")
                sys.exit(0)

            if raw.lower() == "r":
                print("  重新开始…")
                break   # 跳出内层循环，重开一局

            parts = raw.split()
            if len(parts) != 2:
                print("  格式错误，请输入「行 列」，例如：3 4")
                continue

            try:
                r, c = int(parts[0]), int(parts[1])
            except ValueError:
                print("  请输入整数坐标。")
                continue

            if not env._is_valid(r, c):
                print(f"  坐标越界，行 0–{env.rows-1}，列 0–{env.cols-1}。")
                continue

            if env._visible[r][c]:
                print("  该格子已翻开，请选其他格子。")
                continue

            state, reward, done, info = env.step((r, c))

            if done:
                elapsed = time.time() - start_time
                print()
                render_pretty(env, use_color)
                print()
                if info["win"]:
                    session_wins += 1
                    msg = f"  🎉 恭喜！赢了！  翻开 {env._trial_count} 格  用时 {elapsed:.1f}s"
                    print((GREEN + BOLD + msg + RESET) if use_color else msg)
                else:
                    session_losses += 1
                    msg = f"  💥 踩雷了！游戏结束。  翻开 {env._trial_count} 格  用时 {elapsed:.1f}s"
                    print((RED_BG + BOLD + msg + RESET) if use_color else msg)

                total_games = session_wins + session_losses
                print(f"  本次胜负：{session_wins}胜 {session_losses}负  "
                      f"胜率 {session_wins/total_games*100:.0f}%")
                game_over = True

        if game_over:
            try:
                again = input("\n  再来一局？[Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                sys.exit(0)
            if again in ("n", "no"):
                print("感谢游玩！")
                sys.exit(0)


if __name__ == "__main__":
    main()
