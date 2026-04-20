"""
training_evaluation/compare_lambda.py

λ ablation: SARSA(λ) 在不同 λ 下的对比.
共享配置: Decay+Adaptive ε, 9×9/10 雷, 10 维特征.
对比 λ ∈ {0.0, 0.4, 0.8, 1.0}:
    λ=0   -> 单步 SARSA (one-step TD)
    λ=0.4 -> 中等 trace
    λ=0.8 -> 我们的默认值
    λ=1.0 -> 接近 Monte Carlo 行为

产物格式与 compare_epsilon.py / compare_algorithms.py 一致, plot_compare.py 可直接用.

用法:
    python training_evaluation/compare_lambda.py
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
# Decay+Adaptive ε 策略 (复用 compare_algorithms.py 里的)
# ═══════════════════════════════════════════════════════════

class DecayAdaptiveEpsilonStrategy:
    """Decay + Adaptive: 跨局衰减 + 局内 ε_local = ε_base × (1 - opened_ratio)²"""
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
# 训练
# ═══════════════════════════════════════════════════════════

def train_run(seed, lam, args):
    random.seed(seed)
    np.random.seed(seed)

    env = MinesweeperEnv(grid_size=(args.rows, args.cols), num_mines=args.mines)
    strategy = DecayAdaptiveEpsilonStrategy()
    agent = SARSALambdaAgent(alpha=args.alpha, gamma=args.gamma,
                             lam=lam, epsilon=strategy.get_epsilon())

    win_history = []

    for ep in range(args.episodes):
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


# ═══════════════════════════════════════════════════════════
# 评估 (ε=0)
# ═══════════════════════════════════════════════════════════

def evaluate(agent, args, seed):
    random.seed(seed + 10000)
    np.random.seed(seed + 10000)

    env = MinesweeperEnv(grid_size=(args.rows, args.cols), num_mines=args.mines)
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


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rows",  type=int, default=9)
    p.add_argument("--cols",  type=int, default=9)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--episodes",      type=int, default=20000)
    p.add_argument("--test-episodes", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.0, 0.4, 0.8, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--curve-window", type=int, default=500)
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else \
             Path(__file__).resolve().parent / "eval_results" / f"lambda_compare_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "curves").mkdir(exist_ok=True)
    (outdir / "weights").mkdir(exist_ok=True)

    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[lambda] 输出: {outdir}")
    print(f"[lambda] 棋盘: {args.rows}x{args.cols} / {args.mines} 雷")
    print(f"[lambda] λ ∈ {args.lambdas},  seeds={args.seeds}")
    print(f"[lambda] 训练 {args.episodes} 局, 测试 {args.test_episodes} 局\n")

    # λ 命名: 0.0 -> "lambda=0.0", etc.
    def lam_name(lam):
        return f"λ={lam}"

    results = {lam_name(lam): {} for lam in args.lambdas}
    t_start = time.time()

    total_runs = len(args.lambdas) * len(args.seeds)
    run_idx = 0

    for lam in args.lambdas:
        name = lam_name(lam)
        for seed in args.seeds:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {name:<10} seed={seed} ... ",
                  end="", flush=True)
            t0 = time.time()
            agent, win_hist = train_run(seed, lam, args)
            train_time = time.time() - t0

            metrics = evaluate(agent, args, seed)
            results[name][seed] = metrics

            # 训练曲线
            sliding = sliding_win_rate(win_hist, args.curve_window)
            curve_path = outdir / "curves" / f"{name}_seed{seed}.csv"
            with open(curve_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "win", "sliding_win_rate"])
                for i, (won, rate) in enumerate(zip(win_hist, sliding)):
                    w.writerow([i+1, won, f"{rate:.4f}"])

            # 权重
            weight_path = outdir / "weights" / f"{name}_seed{seed}.npz"
            np.savez(weight_path,
                     w=agent.w, lam=lam, seed=seed,
                     episodes=args.episodes)

            print(f"win={metrics['win_rate']*100:5.2f}%  "
                  f"cond_win={metrics['conditional_win_rate']*100:5.2f}%  "
                  f"steps={metrics['avg_steps_all']:5.1f}  "
                  f"ep_ms={metrics['avg_episode_ms']:5.2f}  "
                  f"({train_time:.0f}s)")

    total_time = time.time() - t_start
    print(f"\n[lambda] 全部完成, 总耗时 {total_time/60:.1f} 分钟")

    # 聚合 mean ± std
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
    lines.append(f"λ ablation 实验   棋盘 {args.rows}x{args.cols} / {args.mines} 雷")
    lines.append(f"训练 {args.episodes} 局, 测试 {args.test_episodes} 局, seeds={args.seeds}")
    lines.append(f"共享: SARSA(λ), Decay+Adaptive ε, α={args.alpha}, γ={args.gamma}, 10 维特征")
    lines.append("")

    lines.append("【表 1】胜率 & 步数")
    h1 = f"{'λ':<10}{'Win Rate':>18}{'Steps(All)':>18}{'Steps(Win)':>18}{'Steps(Loss)':>18}{'Coverage':>15}"
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
        lines.append(f"{name:<10}{wr:>18}{sa:>18}{sw:>18}{sl:>18}{cov:>15}")
    lines.append("=" * len(h1))
    lines.append("")

    lines.append("【表 2】测试时间 (ε=0 纯贪婪)")
    h2 = f"{'λ':<10}{'Episode (ms)':>20}{'Per-step (μs)':>20}"
    lines.append("=" * len(h2))
    lines.append(h2)
    lines.append("=" * len(h2))
    for name, s in summary.items():
        em = f"{s['avg_episode_ms_mean']:6.2f} ± {s['avg_episode_ms_std']:5.2f}"
        sm = f"{s['avg_step_us_mean']:6.1f} ± {s['avg_step_us_std']:5.1f}"
        lines.append(f"{name:<10}{em:>20}{sm:>20}")
    lines.append("=" * len(h2))
    lines.append("")

    lines.append("【表 3】首步运气分离")
    h3 = f"{'λ':<10}{'Win Rate':>18}{'First-Click Death':>22}{'Conditional Win Rate':>25}"
    lines.append("=" * len(h3))
    lines.append(h3)
    lines.append("=" * len(h3))
    for name, s in summary.items():
        wr   = f"{s['win_rate_mean']*100:5.2f}% ± {s['win_rate_std']*100:4.2f}"
        fcd  = f"{s['first_click_death_rate_mean']*100:5.2f}% ± {s['first_click_death_rate_std']*100:4.2f}"
        cwr  = f"{s['conditional_win_rate_mean']*100:5.2f}% ± {s['conditional_win_rate_std']*100:4.2f}"
        lines.append(f"{name:<10}{wr:>18}{fcd:>22}{cwr:>25}")
    lines.append("=" * len(h3))

    table = "\n".join(lines)
    print("\n" + table)
    with open(outdir / "summary.txt", "w") as f:
        f.write(table + "\n")

    print(f"\n[lambda] 产物: {outdir}")


if __name__ == "__main__":
    main()