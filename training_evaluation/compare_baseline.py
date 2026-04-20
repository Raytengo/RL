"""
training_evaluation/compare_baseline.py

Random baseline 对比实验。
----------------------------
不训练任何 agent, 只是每步从合法动作里纯随机选, 记录所有指标。
产物格式和 compare_epsilon.py / compare_algorithms.py 完全一致, 方便画图。

用法:
    python training_evaluation/compare_baseline.py
    python training_evaluation/compare_baseline.py --test-episodes 5000

产物: training_evaluation/eval_results/baseline_compare_<timestamp>/
    - config.json
    - results.json
    - summary.txt
    - curves/  (空目录,random 不涉及训练曲线)
    - weights/ (空目录,random 没有权重)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment import MinesweeperEnv


class RandomAgent:
    """纯随机 agent, 无学习, 无权重, 每步从 valid_actions 里均匀随机选."""
    def __init__(self):
        self.epsilon = 0.0  # 接口一致性

    def select_action(self, state, valid_actions):
        return random.choice(valid_actions)


def evaluate_random(args, seed):
    """跑 args.test_episodes 局纯随机, 返回指标字典."""
    random.seed(seed + 10000)
    np.random.seed(seed + 10000)

    env = MinesweeperEnv(grid_size=(args.rows, args.cols), num_mines=args.mines)
    agent = RandomAgent()

    wins = 0
    first_click_deaths = 0
    steps_all = []
    steps_win, steps_loss = [], []
    coverages = []
    episode_times_ms = []
    step_times_us = []
    total_safe = env.rows * env.cols - env.num_mines

    for _ in range(args.test_episodes):
        state = env.reset()
        steps = 0
        ep_t0 = time.perf_counter()

        while not state["done"]:
            valid = env.get_valid_actions()
            if not valid:
                break
            step_t0 = time.perf_counter()
            action = agent.select_action(state, valid)
            step_times_us.append((time.perf_counter() - step_t0) * 1e6)

            state, reward, done, info = env.step(action)
            steps += 1

        episode_times_ms.append((time.perf_counter() - ep_t0) * 1000)
        steps_all.append(steps)

        won = info.get("win", False)
        if won:
            wins += 1
            steps_win.append(steps)
        else:
            steps_loss.append(steps)
            if steps == 1:
                first_click_deaths += 1
        coverages.append(state["trial_count"] / total_safe if total_safe > 0 else 0)

    non_first_click_deaths = args.test_episodes - first_click_deaths
    conditional_win_rate = wins / non_first_click_deaths if non_first_click_deaths > 0 else 0.0

    return {
        "win_rate":              wins / args.test_episodes,
        "conditional_win_rate":  conditional_win_rate,
        "first_click_deaths":    first_click_deaths,
        "first_click_death_rate": first_click_deaths / args.test_episodes,
        "avg_steps_all":         float(np.mean(steps_all)),
        "avg_steps_win":         float(np.mean(steps_win))  if steps_win  else 0.0,
        "avg_steps_loss":        float(np.mean(steps_loss)) if steps_loss else 0.0,
        "avg_coverage":          float(np.mean(coverages)),
        "avg_episode_ms":        float(np.mean(episode_times_ms)),
        "avg_step_us":           float(np.mean(step_times_us)) if step_times_us else 0.0,
        "total_wins":            wins,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rows",  type=int, default=9)
    p.add_argument("--cols",  type=int, default=9)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--test-episodes", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else \
             Path(__file__).resolve().parent / "eval_results" / f"baseline_compare_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "curves").mkdir(exist_ok=True)
    (outdir / "weights").mkdir(exist_ok=True)

    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[baseline] 输出目录: {outdir}")
    print(f"[baseline] 棋盘: {args.rows}x{args.cols}, {args.mines} 雷")
    print(f"[baseline] 测试 {args.test_episodes} 局 × {len(args.seeds)} seeds\n")

    results = {"Random": {}}
    t_start = time.time()

    for i, seed in enumerate(args.seeds, 1):
        t0 = time.time()
        print(f"[{i}/{len(args.seeds)}] Random seed={seed} ... ", end="", flush=True)
        metrics = evaluate_random(args, seed)
        results["Random"][seed] = metrics
        print(f"win_rate={metrics['win_rate']*100:5.2f}%  "
              f"first_click_death={metrics['first_click_death_rate']*100:5.2f}%  "
              f"cond_win={metrics['conditional_win_rate']*100:5.2f}%  "
              f"({time.time()-t0:.1f}s)")

    total_time = time.time() - t_start
    print(f"\n[baseline] 完成, 总耗时 {total_time:.1f}s")

    # 聚合
    seed_results = results["Random"]
    wr       = [r["win_rate"]             for r in seed_results.values()]
    cwr      = [r["conditional_win_rate"] for r in seed_results.values()]
    fcd      = [r["first_click_deaths"]   for r in seed_results.values()]
    fcdr     = [r["first_click_death_rate"] for r in seed_results.values()]
    all_st   = [r["avg_steps_all"]        for r in seed_results.values()]
    sw       = [r["avg_steps_win"]        for r in seed_results.values() if r["avg_steps_win"] > 0]
    sl       = [r["avg_steps_loss"]       for r in seed_results.values()]
    cov      = [r["avg_coverage"]         for r in seed_results.values()]
    ep_ms    = [r["avg_episode_ms"]       for r in seed_results.values()]
    step_us  = [r["avg_step_us"]          for r in seed_results.values()]

    summary = {
        "Random": {
            "win_rate_mean":       float(np.mean(wr)),
            "win_rate_std":        float(np.std(wr)),
            "conditional_win_rate_mean":   float(np.mean(cwr)),
            "conditional_win_rate_std":    float(np.std(cwr)),
            "first_click_deaths_mean":     float(np.mean(fcd)),
            "first_click_death_rate_mean": float(np.mean(fcdr)),
            "first_click_death_rate_std":  float(np.std(fcdr)),
            "avg_steps_all_mean":  float(np.mean(all_st)),
            "avg_steps_all_std":   float(np.std(all_st)),
            "avg_steps_win_mean":  float(np.mean(sw)) if sw else 0.0,
            "avg_steps_win_std":   float(np.std(sw))  if sw else 0.0,
            "avg_steps_loss_mean": float(np.mean(sl)),
            "avg_steps_loss_std":  float(np.std(sl)),
            "avg_coverage_mean":   float(np.mean(cov)),
            "avg_coverage_std":    float(np.std(cov)),
            "avg_episode_ms_mean": float(np.mean(ep_ms)),
            "avg_episode_ms_std":  float(np.std(ep_ms)),
            "avg_step_us_mean":    float(np.mean(step_us)),
            "avg_step_us_std":     float(np.std(step_us)),
            "seeds":               args.seeds,
        }
    }

    with open(outdir / "results.json", "w") as f:
        json.dump({"per_seed": results, "summary": summary},
                  f, indent=2, ensure_ascii=False)

    # 可读表格
    lines = []
    lines.append(f"Random baseline   棋盘 {args.rows}x{args.cols} / {args.mines} 雷")
    lines.append(f"测试 {args.test_episodes} 局, seeds={args.seeds}")
    lines.append("")

    s = summary["Random"]
    lines.append("【Random baseline 指标】")
    lines.append(f"  Win rate:              {s['win_rate_mean']*100:5.2f}% ± {s['win_rate_std']*100:.2f}")
    lines.append(f"  Conditional win rate:  {s['conditional_win_rate_mean']*100:5.2f}% ± {s['conditional_win_rate_std']*100:.2f}")
    lines.append(f"  First-click death:     {s['first_click_death_rate_mean']*100:5.2f}% ± {s['first_click_death_rate_std']*100:.2f}")
    lines.append(f"  Avg steps (all):       {s['avg_steps_all_mean']:.1f} ± {s['avg_steps_all_std']:.1f}")
    lines.append(f"  Avg coverage:          {s['avg_coverage_mean']*100:.1f}% ± {s['avg_coverage_std']*100:.1f}")
    lines.append(f"  Per-episode time:      {s['avg_episode_ms_mean']:.2f} ms ± {s['avg_episode_ms_std']:.2f}")
    lines.append(f"  Per-step time:         {s['avg_step_us_mean']:.1f} μs ± {s['avg_step_us_std']:.1f}")

    table = "\n".join(lines)
    print("\n" + table)
    with open(outdir / "summary.txt", "w") as f:
        f.write(table + "\n")

    print(f"\n[baseline] 产物: {outdir}")


if __name__ == "__main__":
    main()