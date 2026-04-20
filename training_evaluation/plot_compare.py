"""
training_evaluation/plot_compare.py

通用画图脚本，能处理 compare_epsilon.py 或 compare_algorithms.py 的产物。
自动根据 results.json 识别是 ε 对比还是算法对比。

产出 4 张图：
  1. training_curves.png     — 滑动胜率训练曲线 (mean ± std over seeds)
  2. final_win_rate.png      — 最终测试胜率柱状图 (带误差棒)
  3. accuracy_vs_time.png    — accuracy-time trade-off 散点图
  4. multi_metric_bars.png   — 多指标条形图 (胜率/步数/时间归一化)

用法:
    python training_evaluation/plot_compare.py \
        --run training_evaluation/eval_results/eps_compare_20260419_155937

    python training_evaluation/plot_compare.py \
        --run training_evaluation/eval_results/algo_compare_decay_adaptive_20260420_000535
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════
# 配色
# ═══════════════════════════════════════════════════════════

EPS_ORDER   = ["Fixed", "Decay", "Adaptive", "Decay+Adaptive"]
EPS_COLORS  = {
    "Fixed":          "#888780",
    "Decay":          "#185FA5",
    "Adaptive":       "#993C1D",
    "Decay+Adaptive": "#534AB7",
}

ALGO_ORDER  = ["SARSA", "Q-Learning", "MonteCarlo"]
ALGO_COLORS = {
    "SARSA":      "#185FA5",
    "Q-Learning": "#993C1D",
    "MonteCarlo": "#1D9E75",
}


def detect_experiment_type(run_dir: Path):
    json_path = run_dir / "results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"找不到 {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    keys = set(data.get("summary", {}).keys())
    if keys & set(EPS_ORDER):
        return "epsilon", EPS_ORDER, EPS_COLORS, "ε-strategy"
    elif keys & set(ALGO_ORDER):
        return "algorithm", ALGO_ORDER, ALGO_COLORS, "Algorithm"
    else:
        order = list(keys)
        palette = ["#185FA5", "#993C1D", "#1D9E75", "#534AB7",
                   "#888780", "#BA7517", "#D4537E"]
        colors = {n: palette[i % len(palette)] for i, n in enumerate(order)}
        return "unknown", order, colors, "Group"


def load_curves(run_dir: Path, group_order: list):
    curves_dir = run_dir / "curves"
    if not curves_dir.exists():
        raise FileNotFoundError(f"找不到 curves 目录: {curves_dir}")

    data = {g: {} for g in group_order}
    for csv_path in sorted(curves_dir.glob("*.csv")):
        stem = csv_path.stem
        if "_seed" not in stem:
            continue
        group_name, seed_str = stem.rsplit("_seed", 1)
        try:
            seed = int(seed_str)
        except ValueError:
            continue

        rates = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rates.append(float(row["sliding_win_rate"]))

        if group_name in data:
            data[group_name][seed] = np.array(rates)

    for g in group_order:
        if data[g]:
            seeds = sorted(data[g].keys())
            print(f"[load] {g}: seeds={seeds}, length={len(data[g][seeds[0]])}")
        else:
            print(f"[warn] {g}: 没有 seed 数据")
    return data


def load_summary(run_dir: Path):
    with open(run_dir / "results.json", "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════
# 图 1: 训练曲线
# ═══════════════════════════════════════════════════════════

def plot_training_curves(data, group_order, group_colors, out_path: Path,
                         title_prefix: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in group_order:
        if not data.get(name):
            continue
        seed_curves = list(data[name].values())
        min_len = min(len(c) for c in seed_curves)
        stacked = np.stack([c[:min_len] for c in seed_curves])
        mean = stacked.mean(axis=0) * 100
        std  = stacked.std(axis=0)  * 100
        x = np.arange(1, len(mean) + 1)

        color = group_colors.get(name, "#666666")
        ax.plot(x, mean, color=color, linewidth=2.2,
                label=f"{name}  (final={mean[-1]:.1f}%)", zorder=3)
        ax.fill_between(x, mean - std, mean + std,
                        color=color, alpha=0.18, linewidth=0, zorder=2)

    ax.set_xlabel("Training episode", fontsize=12)
    ax.set_ylabel("Sliding win rate (%)", fontsize=12)
    ax.set_title(f"{title_prefix} training curves  (window=500, mean ± 1 std)",
                 fontsize=13, fontweight="500")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.set_ylim(0, max(ax.get_ylim()[1], 75))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] 训练曲线: {out_path}")


# ═══════════════════════════════════════════════════════════
# 图 2: 最终胜率柱状图
# ═══════════════════════════════════════════════════════════

def plot_final_bars(json_data, group_order, group_colors, out_path: Path,
                    title_prefix: str):
    summary = json_data.get("summary", {})
    names, means, stds = [], [], []
    for name in group_order:
        if name not in summary:
            continue
        s = summary[name]
        names.append(name)
        means.append(s["win_rate_mean"] * 100)
        stds.append(s["win_rate_std"] * 100)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(names))
    colors = [group_colors.get(n, "#666666") for n in names]
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, edgecolor="none",
                  error_kw={"elinewidth": 1.4, "capthick": 1.4, "ecolor": "#333"})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 1,
                f"{m:.2f}%\n±{s:.2f}%",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Test win rate (%)", fontsize=12)
    ax.set_title(f"{title_prefix}: final test win rate  "
                 f"(ε=0 greedy, 1000 × 3 seeds)",
                 fontsize=13, fontweight="500")
    ax.set_ylim(0, max(means) + max(stds) + 10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] 胜率柱状图: {out_path}")


# ═══════════════════════════════════════════════════════════
# 图 3: Accuracy-Time 散点图
# ═══════════════════════════════════════════════════════════

def plot_accuracy_vs_time(json_data, group_order, group_colors, out_path: Path,
                          title_prefix: str):
    """
    x 轴: 每局推理耗时 (ms)
    y 轴: 测试胜率 (%)
    每个点的大小: win_rate_std (越大表示越不稳定)
    """
    summary = json_data.get("summary", {})
    rows = []
    for name in group_order:
        if name not in summary:
            continue
        s = summary[name]
        # 有的旧 run 没有 timing 字段, 跳过
        if "avg_episode_ms_mean" not in s:
            print(f"[warn] {name} 缺 avg_episode_ms_mean, 散点图跳过此点")
            continue
        rows.append({
            "name": name,
            "x":    s["avg_episode_ms_mean"],
            "y":    s["win_rate_mean"] * 100,
            "y_err": s["win_rate_std"] * 100,
            "x_err": s.get("avg_episode_ms_std", 0),
        })

    if not rows:
        print("[warn] 所有策略都缺 timing 数据, 无法画 accuracy-time 图")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for r in rows:
        color = group_colors.get(r["name"], "#666666")
        ax.errorbar(r["x"], r["y"],
                    xerr=r["x_err"], yerr=r["y_err"],
                    fmt='o', markersize=14,
                    color=color, ecolor=color, elinewidth=1.2, capsize=4,
                    markeredgecolor="white", markeredgewidth=1.5,
                    alpha=0.9, zorder=3)
        # 标签放在点旁边
        ax.annotate(r["name"],
                    (r["x"], r["y"]),
                    textcoords="offset points",
                    xytext=(12, 6),
                    fontsize=11, fontweight="500")

    # 理想方向箭头: 左上方 = 又快又准
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    x_range = max(xs) - min(xs) + 1e-6
    y_range = max(ys) - min(ys) + 1e-6
    ax.set_xlim(min(xs) - 0.25 * x_range, max(xs) + 0.35 * x_range)
    ax.set_ylim(min(ys) - 0.2 * y_range, max(ys) + 0.2 * y_range)

    # 用 axes 坐标系(0~1 相对于 axis 四角)画箭头
    # 这样箭头永远相对 axis 的四角定位, 不受数据分布窄/宽影响
    ax.annotate("", xy=(0.08, 0.92), xytext=(0.92, 0.08),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#CCCCCC",
                                linewidth=2, alpha=0.6))
    ax.text(0.08, 0.93, " better",
            transform=ax.transAxes,
            fontsize=10, color="#999999", style="italic",
            ha="left", va="bottom")
    ax.text(0.92, 0.07, "worse ",
            transform=ax.transAxes,
            fontsize=10, color="#999999", style="italic",
            ha="right", va="top")

    ax.set_xlabel("Per-episode inference time (ms)", fontsize=12)
    ax.set_ylabel("Test win rate (%)", fontsize=12)
    ax.set_title(f"{title_prefix}: accuracy vs inference time",
                 fontsize=13, fontweight="500")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] accuracy-time 散点: {out_path}")


# ═══════════════════════════════════════════════════════════
# 图 4: 多指标条形图
# ═══════════════════════════════════════════════════════════

def plot_multi_metric_bars(json_data, group_order, group_colors, out_path: Path,
                           title_prefix: str):
    """
    并排三组条形图, 每组对应一个指标:
      - Win Rate (%)
      - Avg steps (all episodes)
      - Per-episode time (ms)
    每个指标下, 每个策略一根柱子。
    """
    summary = json_data.get("summary", {})
    names = [n for n in group_order if n in summary]
    if not names:
        print("[warn] summary 为空, 多指标图跳过")
        return

    # 三个指标的数据
    metric_specs = [
        ("Win rate (%)",  "win_rate_mean",        "win_rate_std",
         lambda v: v * 100, True, True),   # (label, mean_key, std_key, transform, higher_is_better, show)
        ("Avg steps",     "avg_steps_all_mean",   "avg_steps_all_std",
         lambda v: v, None, True),         # 中性: 越大意味着玩得久
        ("Episode time (ms)", "avg_episode_ms_mean", "avg_episode_ms_std",
         lambda v: v, False, True),
    ]

    # 检查哪些指标可用
    available = []
    for label, mk, sk, tf, hib, _ in metric_specs:
        if all(mk in summary.get(n, {}) for n in names):
            available.append((label, mk, sk, tf, hib))

    if not available:
        print("[warn] 没有任何指标在所有策略里都有, 跳过多指标图")
        return

    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5.5))
    if n_metrics == 1:
        axes = [axes]

    for ax, (label, mk, sk, tf, hib) in zip(axes, available):
        means = [tf(summary[n][mk]) for n in names]
        stds  = [tf(summary[n][sk]) if sk in summary[n] else 0 for n in names]
        colors = [group_colors.get(n, "#666666") for n in names]
        x = np.arange(len(names))

        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors, edgecolor="none",
                      error_kw={"elinewidth": 1.2, "capthick": 1.2,
                                "ecolor": "#333"})

        # 柱顶标数字
        for bar, m, s in zip(bars, means, stds):
            offset = max(stds) * 1.1 if max(stds) > 0 else max(means) * 0.02
            ax.text(bar.get_x() + bar.get_width()/2, m + s + offset * 0.3,
                    f"{m:.2f}",
                    ha="center", va="bottom", fontsize=9)

        # 在标题上加个方向箭头, 暗示"越高越好" / "越低越好"
        if hib is True:
            arrow = " ↑"
        elif hib is False:
            arrow = " ↓"
        else:
            arrow = ""

        ax.set_title(label + arrow, fontsize=12, fontweight="500")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        # y 轴留点空间给数字
        y_max = max(means) + max(stds) if max(stds) > 0 else max(means)
        ax.set_ylim(0, y_max * 1.18)

    fig.suptitle(f"{title_prefix}: multi-metric comparison",
                 fontsize=14, fontweight="500", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 多指标条形图: {out_path}")


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=str)
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run).resolve()
    if not run_dir.exists():
        print(f"[err] 找不到 run 目录: {run_dir}")
        sys.exit(1)

    outdir = Path(args.outdir).resolve() if args.outdir else run_dir
    outdir.mkdir(parents=True, exist_ok=True)

    exp_type, group_order, group_colors, title_prefix = detect_experiment_type(run_dir)
    print(f"[plot] 实验类型: {exp_type}  (title='{title_prefix}')")
    print(f"[plot] 输出到: {outdir}\n")

    print("[plot] 读取 curves/...")
    data = load_curves(run_dir, group_order)
    json_data = load_summary(run_dir)
    print()

    # 生成 4 张图
    plot_training_curves(data, group_order, group_colors,
                         outdir / "training_curves.png", title_prefix)
    plot_final_bars(json_data, group_order, group_colors,
                    outdir / "final_win_rate.png", title_prefix)
    plot_accuracy_vs_time(json_data, group_order, group_colors,
                          outdir / "accuracy_vs_time.png", title_prefix)
    plot_multi_metric_bars(json_data, group_order, group_colors,
                           outdir / "multi_metric_bars.png", title_prefix)

    print("\n[plot] 全部完成")


if __name__ == "__main__":
    main()