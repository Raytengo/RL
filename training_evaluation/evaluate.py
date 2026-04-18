"""
training_evaluation/evaluate.py

评估已训练模型（贪婪策略，epsilon=0），输出多局统计指标。

用法示例：
  python -m training_evaluation.evaluate \
        --episodes 1000

    # 指定某个 run 目录
    python -m training_evaluation.evaluate \
        --run training_evaluation/runs/20260417_125806_9x9_m10_reward_report
"""

import argparse
import json
import sys
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
    parser = argparse.ArgumentParser(description="评估已训练模型")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="训练产物目录（需包含 config.json 和 final_model.npz）",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="评估局数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="单局最多步数，防止意外死循环",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="将评估结果保存到 run 目录下 evaluate_report.md",
    )
    parser.add_argument(
        "--no-save-report",
        action="store_false",
        dest="save_report",
        help="只在终端输出，不保存评估报告",
    )
    return parser.parse_args()


def find_latest_run_dir(runs_dir: Path) -> Path:
    if not runs_dir.exists():
        raise FileNotFoundError(f"找不到 runs 目录: {runs_dir}")

    candidates = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        if (p / "config.json").exists() and (p / "final_model.npz").exists():
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"在 {runs_dir} 下找不到可评估的 run 目录")

    # 优先按修改时间，时间相同再按目录名，确保结果稳定。
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)[0]


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


def run_one_episode(env: MinesweeperEnv, agent: SARSALambdaAgent, max_steps: int):
    state = env.reset()
    done = False
    steps = 0
    total_reward = 0.0

    while not done and steps < max_steps:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        action = agent.select_action(state, valid_actions)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    return {
        "win": int(state.get("win", False)),
        "reward": float(total_reward),
        "steps": steps,
        "trial_count": int(state.get("trial_count", 0)),
    }


def summarize(stats_list, window=500):
    wins = [s["win"] for s in stats_list]
    rewards = [s["reward"] for s in stats_list]
    steps = [s["steps"] for s in stats_list]
    trials = [s["trial_count"] for s in stats_list]

    total = len(stats_list)
    total_wins = int(sum(wins))
    total_losses = total - total_wins

    win_rewards = [s["reward"] for s in stats_list if s["win"] == 1]
    loss_rewards = [s["reward"] for s in stats_list if s["win"] == 0]
    win_steps = [s["steps"] for s in stats_list if s["win"] == 1]
    loss_steps = [s["steps"] for s in stats_list if s["win"] == 0]
    win_trials = [s["trial_count"] for s in stats_list if s["win"] == 1]
    loss_trials = [s["trial_count"] for s in stats_list if s["win"] == 0]

    def avg(values):
        return float(np.mean(values)) if values else 0.0

    tail_n = min(window, total)
    tail_win_rate = sum(wins[-tail_n:]) / tail_n if tail_n > 0 else 0.0

    return {
        "episodes": total,
        "success_count": total_wins,
        "failure_count": total_losses,
        "overall_win_rate": total_wins / total if total > 0 else 0.0,
        "tail_window": tail_n,
        "tail_win_rate": tail_win_rate,
        "avg_reward": avg(rewards),
        "avg_steps": avg(steps),
        "avg_opened": avg(trials),
        "avg_reward_win": avg(win_rewards),
        "avg_reward_loss": avg(loss_rewards),
        "avg_steps_win": avg(win_steps),
        "avg_steps_loss": avg(loss_steps),
        "avg_opened_win": avg(win_trials),
        "avg_opened_loss": avg(loss_trials),
    }


def build_report_text(summary):
    lines = [
        "## 模型评估摘要",
        f"- 胜/负：{summary['success_count']} / {summary['failure_count']}（总计 {summary['episodes']} 局）",
        f"- 整体胜率：{summary['overall_win_rate'] * 100:.3f}%",
        f"- 近 {summary['tail_window']} 局胜率：{summary['tail_win_rate'] * 100:.3f}%",
        f"- 平均奖励：{summary['avg_reward']:.4f}",
        f"- 平均步数：{summary['avg_steps']:.4f}",
        f"- 平均翻开格数：{summary['avg_opened']:.4f}",
        "",
        "## 胜负对比",
        f"- 奖励（胜/负）：{summary['avg_reward_win']:.4f} / {summary['avg_reward_loss']:.4f}",
        f"- 步数（胜/负）：{summary['avg_steps_win']:.4f} / {summary['avg_steps_loss']:.4f}",
        f"- 翻开格数（胜/负）：{summary['avg_opened_win']:.4f} / {summary['avg_opened_loss']:.4f}",
    ]
    return "\n".join(lines) + "\n"


def main():
    configure_console_output()
    args = parse_args()
    np.random.seed(args.seed)

    if args.run:
        run_dir = Path(args.run).resolve()
    else:
        runs_dir = Path(__file__).resolve().parent / "runs"
        run_dir = find_latest_run_dir(runs_dir)
        print(f"[evaluate] 未指定 --run，自动使用最新 run: {run_dir}")

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

    stats_list = []
    for _ in range(args.episodes):
        stats_list.append(run_one_episode(env, agent, max_steps=args.max_steps))

    summary = summarize(stats_list)
    report_text = build_report_text(summary)

    print(f"[evaluate] run_dir: {run_dir}")
    print(f"[evaluate] board: {rows}x{cols}, mines={mines}")
    print(report_text)

    if args.save_report:
        report_path = run_dir / "evaluate_report.md"
        report_path.write_text(report_text, encoding="utf-8")
        print(f"[evaluate] 评估报告已保存: {report_path}")


if __name__ == "__main__":
    main()
