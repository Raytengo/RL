"""
training_evaluation/compare_difficulty.py

难度对比: 同一最佳配置 (SARSA λ=0.8 + Decay+Adaptive ε) 在不同棋盘上的表现.
默认对比: 4×4/2 雷, 6×6/5 雷, 9×9/10 雷  (雷密度都接近 12%)

注意: 不同棋盘需要不同的 episode 数才能收敛.
默认按棋盘大小自动调整: 4×4 用 5000 局, 6×6 用 10000 局, 9×9 用 20000 局.
也可以用 --episodes 强制指定统一值.

产物格式与其他 compare_*.py 一致, plot_compare.py 可直接用.

用法:
    python training_evaluation/compare_difficulty.py
    python training_evaluation/compare_difficulty.py --episodes 10000  # 统一 10000 局
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment import MinesweeperEnv
from algorithm.features import extract_features
from algorithm.agent import SARSALambdaAgent


# ═══════════════════════════════════════════════════════════
# Decay+Adaptive ε
# ═══════════════════════════════════════════════════════════

class DecayAdaptiveEpsilonStrategy:
    def __init__(self, eps_start=0.3, eps_end=0.05, eps_decay=0.9995):
        self.eps_base = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def get_epsilon(self, state=None, env=None):
        if state is None or env is None:
            return self.eps_base
        opened_ratio = state["trial_count"] / (env.rows * env.cols)
        return self.eps_base * (1 - opened_ratio) ** 2

    def on_episode_end(self):
        self.eps_base = max(self.eps_end, self.eps_base * self.eps_decay)


# ═══════════════════════════════════════════════════════════
# 默认棋盘配置 (rows, cols, mines, episodes)
# ═══════════════════════════════════════════════════════════

DEFAULT_BOARDS = [
    {"name": "4x4 (2 mines)",  "rows": 4, "cols": 4, "mines": 2,  "episodes": 5000},
    {"name": "6x6 (5 mines)",  "rows": 6, "cols": 6, "mines": 5,  "episodes": 10000},
    {"name": "9x9 (10 mines)", "rows": 9, "cols": 9, "mines": 10, "episodes": 20000},
]


# ═══════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════

def train_run(board, seed, args):
    random.seed(seed)
    np.random.seed(seed)

    env = MinesweeperEnv(grid_size=(board["rows"], board["cols"]),
                         num_mines=board["mines"])
    strategy = DecayAdaptiveEpsilonStrategy()
    agent = SARSALambdaAgent(alpha=args.alpha, gamma=args.gamma,
                             lam=args.lam, epsilon=strategy.get_epsilon())

    win_history = []
    episodes = args.episodes if args.episodes else board["episodes"]

    for ep in range(episodes):
        state = env.reset()
        agent.reset_trace()

        valid = env.get_valid_actions()
        agent.epsilon = strategy.get_epsilon(state, env)
        action = agent.select_action(state, valid)

        while not state["done"]:
            next_state, reward, done, info = env.step(action)
            valid_next = env.get_valid_actions() if not done else []
            agent.epsilon = strategy.get_epsilon(next_state, env)
            next_action = agent.select_action(next_state, valid_next) if valid_next else None
            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            if not valid_next:
                break

        win_history.append(1 if info.get("win") else 0)
        strategy.on_episode_end()

    return agent, win_history


def evaluate(agent, board, args, seed):
    random.seed(seed + 10000)
    np.random.seed(seed + 10000)

    env = MinesweeperEnv(grid_size=(board["rows"], board["cols"]),
                         num_mines=board["mines"])
    agent.epsilon = 0.0

    wins = 0
    first_click_deaths = 0
    steps_all, steps_win, steps_loss, coverages = [], [], [], []
    episode_times_ms, step_times_us = [], []
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
    cond_wr = wins / non_first_click_deaths if non_first_click_deaths > 0 else 0.0

    return {
        "win_rate":              wins / args.test_episodes,
        "conditional_win_rate":  cond_wr,
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


def sliding_win_rate(win_history, window=500):
    result = []
    for i in range(len(win_history)):
        lo = max(0, i + 1 - window)
        result.append(sum(win_history[lo:i+1]) / (i + 1 - lo))
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=0,
                   help="0 = 按棋盘自动 (4x4=5000, 6x6=10000, 9x9=20000)")
    p.add_argument("--test-episodes", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam",   type=float, default=0.8)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--curve-window", type=int, default=500)
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else \
             Path(__file__).resolve().parent / "eval_results" / f"difficulty_compare_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "curves").mkdir(exist_ok=True)
    (outdir / "weights").mkdir(exist_ok=True)

    config = vars(args).copy()
    config["boards"] = DEFAULT_BOARDS
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[difficulty] 输出: {outdir}")
    print(f"[difficulty] 共享配置: SARSA(λ={args.lam}) + Decay+Adaptive ε")
    print(f"[difficulty] seeds={args.seeds}, 测试 {args.test_episodes} 局/棋盘")
    for b in DEFAULT_BOARDS:
        eps = args.episodes if args.episodes else b["episodes"]
        print(f"             {b['name']:<20} 训练 {eps} 局")
    print()

    results = {b["name"]: {} for b in DEFAULT_BOARDS}
    t_start = time.time()
    total_runs = len(DEFAULT_BOARDS) * len(args.seeds)
    run_idx = 0

    for board in DEFAULT_BOARDS:
        name = board["name"]
        for seed in args.seeds:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {name:<20} seed={seed} ... ",
                  end="", flush=True)
            t0 = time.time()
            agent, win_hist = train_run(board, seed, args)
            train_time = time.time() - t0

            metrics = evaluate(agent, board, args, seed)
            results[name][seed] = metrics

            sliding = sliding_win_rate(win_hist, args.curve_window)
            curve_path = outdir / "curves" / f"{name.replace(' ', '_')}_seed{seed}.csv"
            with open(curve_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "win", "sliding_win_rate"])
                for i, (won, rate) in enumerate(zip(win_hist, sliding)):
                    w.writerow([i+1, won, f"{rate:.4f}"])

            weight_path = outdir / "weights" / f"{name.replace(' ', '_')}_seed{seed}.npz"
            np.savez(weight_path, w=agent.w, board=name, seed=seed)

            print(f"win={metrics['win_rate']*100:5.2f}%  "
                  f"cond_win={metrics['conditional_win_rate']*100:5.2f}%  "
                  f"steps={metrics['avg_steps_all']:5.1f}  "
                  f"({train_time:.0f}s)")

    total_time = time.time() - t_start
    print(f"\n[difficulty] 全部完成, 总耗时 {total_time/60:.1f} 分钟")

    # 聚合
    summary = {}
    for name, seed_results in results.items():
        if not seed_results:
            continue
        wr       = [r["win_rate"]              for r in seed_results.values()]
        cwr      = [r["conditional_win_rate"]  for r in seed_results.values()]
        fcd      = [r["first_click_deaths"]    for r in seed_results.values()]
        fcdr     = [r["first_click_death_rate"] for r in seed_results.values()]
        all_st   = [r["avg_steps_all"]         for r in seed_results.values()]
        sw       = [r["avg_steps_win"]         for r in seed_results.values() if r["avg_steps_win"] > 0]
        sl       = [r["avg_steps_loss"]        for r in seed_results.values()]
        cov      = [r["avg_coverage"]          for r in seed_results.values()]
        ep_ms    = [r["avg_episode_ms"]        for r in seed_results.values()]
        step_us  = [r["avg_step_us"]           for r in seed_results.values()]

        summary[name] = {
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

    with open(outdir / "results.json", "w") as f:
        json.dump({"per_seed": results, "summary": summary},
                  f, indent=2, ensure_ascii=False)

    # 表格
    lines = []
    lines.append(f"难度对比实验  (SARSA(λ={args.lam}) + Decay+Adaptive ε)")
    lines.append(f"测试 {args.test_episodes} 局, seeds={args.seeds}")
    lines.append(f"α={args.alpha}, γ={args.gamma}")
    lines.append("")

    lines.append("【表 1】胜率 & 步数")
    h1 = f"{'Board':<22}{'Win Rate':>18}{'Steps(All)':>18}{'Steps(Win)':>18}{'Steps(Loss)':>18}{'Coverage':>15}"
    lines.append("=" * len(h1))
    lines.append(h1)
    lines.append("=" * len(h1))
    for name, s in summary.items():
        wr = f"{s['win_rate_mean']*100:5.2f}% ± {s['win_rate_std']*100:4.2f}"
        sa = f"{s['avg_steps_all_mean']:5.1f} ± {s['avg_steps_all_std']:4.1f}"
        sw = f"{s['avg_steps_win_mean']:5.1f} ± {s['avg_steps_win_std']:4.1f}" \
            if s['avg_steps_win_mean'] > 0 else "N/A"
        sl = f"{s['avg_steps_loss_mean']:5.1f} ± {s['avg_steps_loss_std']:4.1f}"
        cov = f"{s['avg_coverage_mean']*100:5.1f}% ± {s['avg_coverage_std']*100:4.1f}"
        lines.append(f"{name:<22}{wr:>18}{sa:>18}{sw:>18}{sl:>18}{cov:>15}")
    lines.append("=" * len(h1))
    lines.append("")

    lines.append("【表 2】测试时间 (ε=0 纯贪婪)")
    h2 = f"{'Board':<22}{'Episode (ms)':>20}{'Per-step (μs)':>20}"
    lines.append("=" * len(h2))
    lines.append(h2)
    lines.append("=" * len(h2))
    for name, s in summary.items():
        em = f"{s['avg_episode_ms_mean']:6.2f} ± {s['avg_episode_ms_std']:5.2f}"
        sm = f"{s['avg_step_us_mean']:6.1f} ± {s['avg_step_us_std']:5.1f}"
        lines.append(f"{name:<22}{em:>20}{sm:>20}")
    lines.append("=" * len(h2))
    lines.append("")

    lines.append("【表 3】首步运气分离")
    h3 = f"{'Board':<22}{'Win Rate':>18}{'First-Click Death':>22}{'Conditional Win Rate':>25}"
    lines.append("=" * len(h3))
    lines.append(h3)
    lines.append("=" * len(h3))
    for name, s in summary.items():
        wr   = f"{s['win_rate_mean']*100:5.2f}% ± {s['win_rate_std']*100:4.2f}"
        fcd  = f"{s['first_click_death_rate_mean']*100:5.2f}% ± {s['first_click_death_rate_std']*100:4.2f}"
        cwr  = f"{s['conditional_win_rate_mean']*100:5.2f}% ± {s['conditional_win_rate_std']*100:4.2f}"
        lines.append(f"{name:<22}{wr:>18}{fcd:>22}{cwr:>25}")
    lines.append("=" * len(h3))

    table = "\n".join(lines)
    print("\n" + table)
    with open(outdir / "summary.txt", "w") as f:
        f.write(table + "\n")

    print(f"\n[difficulty] 产物: {outdir}")


if __name__ == "__main__":
    main()