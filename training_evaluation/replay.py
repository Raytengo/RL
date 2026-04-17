"""
training_evaluation/replay.py

载入已训练模型并在终端逐步回放模型的扫雷过程。

用法示例：
  python -m training_evaluation.replay \
    --run training_evaluation/runs/20260417_110523_9x9_m10 \
    --episodes 1 --delay 0.15
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment import MinesweeperEnv
from algorithm.agent import SARSALambdaAgent


def configure_console_output():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="回放已训练模型的对局过程")
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="训练产物目录（需包含 config.json 和 final_model.npz）",
    )
    parser.add_argument("--episodes", type=int, default=1, help="回放局数")
    parser.add_argument("--delay", type=float, default=0.15, help="每步停顿秒数")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="单局最多步数，防止意外死循环",
    )
    return parser.parse_args()


def load_run(run_dir: Path):
    config_path = run_dir / "config.json"
    model_path = run_dir / "final_model.npz"

    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    model = np.load(model_path)
    if "w" not in model:
        raise KeyError(f"模型文件缺少权重字段 w: {model_path}")
    return config, model["w"]


def replay_one_episode(env: MinesweeperEnv, agent: SARSALambdaAgent, delay: float, max_steps: int):
    state = env.reset()
    done = False
    step_idx = 0
    total_reward = 0.0

    print("\n[replay] 初始棋盘")
    env.render()

    while not done and step_idx < max_steps:
        valid = env.get_valid_actions()
        if not valid:
            break
        action = agent.select_action(state, valid)
        state, reward, done, info = env.step(action)
        step_idx += 1
        total_reward += reward

        print(
            f"\n[step {step_idx:03d}] action={action} reward={reward:.3f} "
            f"trial_count={state['trial_count']} done={done}"
        )
        env.render()
        if delay > 0:
            time.sleep(delay)

    win = bool(state.get("win", False))
    print(
        f"\n[result] {'WIN' if win else 'LOSE'} | steps={step_idx} "
        f"| opened={state['trial_count']} | total_reward={total_reward:.3f}"
    )
    return win


def main():
    configure_console_output()
    args = parse_args()
    run_dir = Path(args.run).resolve()

    config, w = load_run(run_dir)
    rows = int(config["rows"])
    cols = int(config["cols"])
    mines = int(config["mines"])
    alpha = float(config.get("alpha", 0.01))
    gamma = float(config.get("gamma", 0.99))
    lam = float(config.get("lam", 0.8))
    safe_open_reward_per_cell = float(config.get("safe_open_reward_per_cell", 0.05))
    win_reward = float(config.get("win_reward", 10.0))
    lose_reward = float(config.get("lose_reward", -10.0))
    repeat_reward = float(config.get("repeat_reward", -0.5))

    env = MinesweeperEnv(
        grid_size=(rows, cols),
        num_mines=mines,
        safe_open_reward_per_cell=safe_open_reward_per_cell,
        win_reward=win_reward,
        lose_reward=lose_reward,
        repeat_reward=repeat_reward,
    )
    agent = SARSALambdaAgent(alpha=alpha, gamma=gamma, lam=lam, epsilon=0.0)
    agent.w = np.array(w, dtype=float)

    print(f"[replay] run_dir: {run_dir}")
    print(f"[replay] board: {rows}x{cols}, mines={mines}")
    print(f"[replay] episodes: {args.episodes}, delay={args.delay}s")

    wins = 0
    for ep in range(1, args.episodes + 1):
        print("\n" + "=" * 48)
        print(f"[episode {ep}/{args.episodes}]")
        if replay_one_episode(env, agent, delay=args.delay, max_steps=args.max_steps):
            wins += 1

    print("\n" + "=" * 48)
    print(f"[summary] wins={wins}/{args.episodes} ({wins / args.episodes * 100:.1f}%)")


if __name__ == "__main__":
    main()
