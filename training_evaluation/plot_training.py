"""
Plot training metrics from a saved run directory.

Example:
    python training_evaluation/plot_training.py \
        --run training_evaluation/runs/20260417_110523_9x9_m10
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_LABELS = [
    "f1 hidden_nbr_ratio",
    "f2 number_sum_norm",
    "f3 danger_estimate",
    "f4 isolated_flag",
    "f5 global_hidden_ratio",
    "f6 visible_mine_ratio",
    "f7 bias",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot metrics for one training run")
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run directory containing config.json, log.csv, and final_model.npz",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path or directory, default: <run>/training_summary.png",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=200,
        help="Moving-average window for reward and steps",
    )
    return parser.parse_args()


def load_run_data(run_dir: Path):
    config_path = run_dir / "config.json"
    log_path = run_dir / "log.csv"
    model_path = run_dir / "final_model.npz"

    missing = [p.name for p in (config_path, log_path, model_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run directory missing files: {', '.join(missing)}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    episodes = []
    rewards = []
    steps = []
    with open(log_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            steps.append(float(row["steps"]))

    weights = np.array(np.load(model_path)["w"], dtype=float)
    return config, np.array(episodes), np.array(rewards), np.array(steps), weights


def moving_average(values: np.ndarray, window: int):
    if window <= 1 or len(values) == 0:
        return values
    window = min(window, len(values))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def feature_labels(weight_dim: int):
    if weight_dim == len(FEATURE_LABELS):
        return FEATURE_LABELS
    return [f"f{i + 1}" for i in range(weight_dim)]


def resolve_out_path(run_dir: Path, out_arg: str | None) -> Path:
    if not out_arg:
        return run_dir / "training_summary.png"

    candidate = Path(out_arg).resolve()
    if candidate.suffix:
        return candidate
    return candidate / "training_summary.png"


def save_training_summary_plot(run_dir: Path, out_path: Path, smooth_window: int = 200) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config, episodes, rewards, steps, weights = load_run_data(run_dir)
    labels = feature_labels(len(weights))

    reward_ma = moving_average(rewards, smooth_window)
    step_ma = moving_average(steps, smooth_window)
    ma_episodes = episodes[len(episodes) - len(reward_ma):]

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True)
    fig.suptitle(
        (
            f"Training Summary: {run_dir.name}\n"
            f"board={config.get('rows')}x{config.get('cols')} mines={config.get('mines')} "
            f"episodes={config.get('episodes')}"
        ),
        fontsize=14,
        fontweight="bold",
    )

    bar_colors = ["#2a9d8f" if w >= 0 else "#e76f51" for w in weights]
    axes[0].bar(range(len(weights)), weights, color=bar_colors)
    axes[0].axhline(0.0, color="#222222", linewidth=1)
    axes[0].set_title("Final Feature Weights")
    axes[0].set_ylabel("weight")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    for idx, value in enumerate(weights):
        offset = 0.03 * max(1.0, np.max(np.abs(weights)))
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        axes[0].text(idx, y, f"{value:.2f}", ha="center", va=va, fontsize=9)

    axes[1].plot(episodes, rewards, color="#8ecae6", linewidth=0.8, alpha=0.5, label="reward")
    if len(reward_ma) > 0:
        axes[1].plot(
            ma_episodes,
            reward_ma,
            color="#1d3557",
            linewidth=2.0,
            label=f"moving avg ({min(smooth_window, len(rewards))})",
        )
    axes[1].set_title("Reward per Episode")
    axes[1].set_ylabel("reward")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(episodes, steps, color="#bde0fe", linewidth=0.8, alpha=0.55, label="steps")
    if len(step_ma) > 0:
        axes[2].plot(
            ma_episodes,
            step_ma,
            color="#6d597a",
            linewidth=2.0,
            label=f"moving avg ({min(smooth_window, len(steps))})",
        )
    axes[2].set_title("Steps per Episode")
    axes[2].set_xlabel("episode")
    axes[2].set_ylabel("steps")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    run_dir = Path(args.run).resolve()
    out_path = resolve_out_path(run_dir, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    saved_path = save_training_summary_plot(
        run_dir=run_dir,
        out_path=out_path,
        smooth_window=args.smooth_window,
    )
    print(f"saved_plot={saved_path}")


if __name__ == "__main__":
    main()
