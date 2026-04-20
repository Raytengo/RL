"""
training_evaluation/compare_epsilon.py

ε-策略对比实验: 2x2 ablation
--------------------------------
变量控制:
  - 共享的 features.py (10 维) 和 MinesweeperEnv (不改)
  - 共享超参: alpha=0.01, gamma=0.99, lam=0.8, episodes=20000
  - 共享 seed 集合 (默认 42, 43, 44, 对每个策略都跑一遍取均值)
  - 唯一变量: epsilon 策略

2x2 Ablation 设计:
                  | 跨局不衰减 |  跨局衰减 (0.3→0.05, decay=0.9995)
  ----------------|-----------|-----------------------------------
  局内不自适应     |  Fixed    |  Decay            <- 基准组, 和 train.py 默认一致
  局内自适应 (B)  |  Adaptive |  Decay+Adaptive

Adaptive 公式 (方案 B, 幂函数):
    opened_ratio = state["trial_count"] / (rows * cols)
    eps_local = eps_base * (1 - opened_ratio) ** 2

用法:
    # 完整跑 (4 策略 x 3 seed x 20000 局, 约 20 分钟)
    python training_evaluation/compare_epsilon.py

    # 快速 smoke test (先确认能跑通)
    python3 training_evaluation/compare_epsilon.py --episodes 500 --seeds 42 --test-episodes 200

产物: training_evaluation/eval_results/eps_compare_<timestamp>/
    - results.json     每个 (策略, seed) 的完整指标
    - summary.txt      可读的对比表 (均值 ± 标准差)
    - curves/*.csv     每组的滑动胜率曲线, 方便画图
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
from algorithm.agent import SARSALambdaAgent
from algorithm.features import extract_features, FEATURE_DIM


# ═══════════════════════════════════════════════════════════
# 四种 ε 策略
# ═══════════════════════════════════════════════════════════

class EpsilonStrategy:
    """
    每种策略需要实现两个方法:
      - get_epsilon(state, env) -> float
            每一步调用, 返回当前决策要用的 ε
      - on_episode_end()
            每局结束后调用, 用于跨局衰减
    """

    def __init__(self, name, eps_start=0.3, eps_end=0.05,
                 eps_decay=0.9995, adaptive=False):
        self.name = name
        self.eps_base = eps_start       # 当前"基准" ε (可能随 episode 衰减)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.adaptive = adaptive        # 是否按局内已翻开比例缩放
        self.do_decay = eps_decay < 1.0 # 是否跨局衰减

    def get_epsilon(self, state, env):
        eps = self.eps_base
        if self.adaptive:
            opened_ratio = state["trial_count"] / (env.rows * env.cols)
            eps = eps * (1 - opened_ratio) ** 2   # 方案 B: 幂函数
        return eps

    def on_episode_end(self):
        if self.do_decay:
            self.eps_base = max(self.eps_end, self.eps_base * self.eps_decay)


def build_strategies():
    """返回 4 种策略的配置 (名字 + 构造参数)"""
    return {
        # 局内不变, 跨局不衰减
        "Fixed":           dict(eps_start=0.1, eps_end=0.1,  eps_decay=1.0,    adaptive=False),
        # 局内不变, 跨局衰减 <- 和 train.py 默认完全一致 (基准组)
        "Decay":           dict(eps_start=0.3, eps_end=0.05, eps_decay=0.9995, adaptive=False),
        # 局内自适应, 跨局不衰减
        "Adaptive":        dict(eps_start=0.3, eps_end=0.3,  eps_decay=1.0,    adaptive=True),
        # 局内自适应 + 跨局衰减 (两者叠加)
        "Decay+Adaptive":  dict(eps_start=0.3, eps_end=0.05, eps_decay=0.9995, adaptive=True),
    }


# ═══════════════════════════════════════════════════════════
# 训练一次 (给定策略 + seed)
# ═══════════════════════════════════════════════════════════

def train_one_run(strategy_name, strategy_cfg, seed, args):
    """训练一个 agent, 返回 (agent, win_history)"""
    random.seed(seed)
    np.random.seed(seed)

    env = MinesweeperEnv(grid_size=(args.rows, args.cols), num_mines=args.mines)
    strategy = EpsilonStrategy(strategy_name, **strategy_cfg)

    # 注意: agent 的 epsilon 属性会被每步动态覆盖, 初值随便填
    agent = SARSALambdaAgent(
        alpha=args.alpha, gamma=args.gamma, lam=args.lam,
        epsilon=strategy.eps_base,
    )

    win_history = []  # 每局胜负, 1 或 0

    for ep in range(args.episodes):
        state = env.reset()
        agent.reset_trace()
        valid = env.get_valid_actions()

        # 每一步根据策略动态设置 agent.epsilon
        agent.epsilon = strategy.get_epsilon(state, env)
        action = agent.select_action(state, valid)

        while True:
            next_state, reward, done, info = env.step(action)
            if done:
                agent.update(state, action, reward, None, None, done=True)
                break
            next_valid = env.get_valid_actions()
            agent.epsilon = strategy.get_epsilon(next_state, env)
            next_action = agent.select_action(next_state, next_valid)
            agent.update(state, action, reward, next_state, next_action, done=False)
            state, action = next_state, next_action

        win_history.append(1 if info.get("win") else 0)
        strategy.on_episode_end()

    return agent, win_history


# ═══════════════════════════════════════════════════════════
# 评估 (ε=0 纯贪婪)
# ═══════════════════════════════════════════════════════════

def evaluate(agent, args, seed):
    """
    用 ε=0 评估, 返回指标字典.
    记录: 胜率 / 条件胜率 / 首步踩雷率 / 步数 / 时间.
    """
    random.seed(seed + 10000)   # 评估用独立 seed, 避免训练/评估撞种子
    np.random.seed(seed + 10000)

    env = MinesweeperEnv(grid_size=(args.rows, args.cols), num_mines=args.mines)
    agent.epsilon = 0.0   # 纯贪婪

    wins = 0
    first_click_deaths = 0   # 第一步就踩雷的局数 (纯运气, 不怪 agent)
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
            # 只计时 select_action 的耗时, 不含 env.step
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
            # 第一步就挂 = 开局无任何信息, 纯运气
            if steps == 1:
                first_click_deaths += 1
        coverages.append(state["trial_count"] / total_safe if total_safe > 0 else 0)

    # 条件胜率: 排除首步即死的局后的胜率 ("纯 skill 胜率")
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


# ═══════════════════════════════════════════════════════════
# 滑动胜率 (用于训练曲线)
# ═══════════════════════════════════════════════════════════

def sliding_win_rate(win_history, window=500):
    """返回每一 episode 的滑动胜率 (用于画训练曲线)"""
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
    # 环境
    p.add_argument("--rows",  type=int, default=9)
    p.add_argument("--cols",  type=int, default=9)
    p.add_argument("--mines", type=int, default=10)
    # 训练
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--alpha",    type=float, default=0.01)
    p.add_argument("--gamma",    type=float, default=0.99)
    p.add_argument("--lam",      type=float, default=0.8)
    # 评估
    p.add_argument("--test-episodes", type=int, default=1000)
    # seed
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    # 输出
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--curve-window", type=int, default=500,
                   help="训练曲线的滑动胜率窗口")
    return p.parse_args()


def main():
    args = parse_args()

    # 输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else \
             Path(__file__).resolve().parent / "eval_results" / f"eps_compare_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "curves").mkdir(exist_ok=True)
    (outdir / "weights").mkdir(exist_ok=True)

    # 保存本次实验的配置
    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    strategies = build_strategies()
    print(f"[compare] 输出目录: {outdir}")
    print(f"[compare] 棋盘: {args.rows}x{args.cols}, {args.mines} 雷")
    print(f"[compare] 训练 {args.episodes} 局, 评估 {args.test_episodes} 局")
    print(f"[compare] Seeds: {args.seeds}")
    print(f"[compare] 策略: {list(strategies.keys())}")
    print(f"[compare] 注: 'Decay' 策略 + seed=42 对应 train.py 默认配置\n")

    # results[strategy][seed] = {指标字典}
    results = {name: {} for name in strategies}
    total_runs = len(strategies) * len(args.seeds)
    run_idx = 0
    t_start = time.time()

    for strat_name, strat_cfg in strategies.items():
        for seed in args.seeds:
            run_idx += 1
            t0 = time.time()
            print(f"[{run_idx:2d}/{total_runs}] {strat_name:<20s} seed={seed} ... ", end="", flush=True)

            agent, win_history = train_one_run(strat_name, strat_cfg, seed, args)
            train_time = time.time() - t0

            metrics = evaluate(agent, args, seed)
            metrics["train_time_sec"] = train_time
            results[strat_name][seed] = metrics

            # 保存训练曲线
            curve = sliding_win_rate(win_history, window=args.curve_window)
            curve_path = outdir / "curves" / f"{strat_name}_seed{seed}.csv"
            with open(curve_path, "w", newline="") as cf:
                w = csv.writer(cf)
                w.writerow(["episode", "win", "sliding_win_rate"])
                for i, (win, rate) in enumerate(zip(win_history, curve), 1):
                    w.writerow([i, win, f"{rate:.4f}"])

            # 保存 agent 权重, 方便后续 benchmark / 重新评估
            weight_path = outdir / "weights" / f"{strat_name}_seed{seed}.npz"
            np.savez(weight_path,
                     w=agent.w,
                     strategy=strat_name,
                     seed=seed,
                     episodes=args.episodes)

            print(f"win_rate={metrics['win_rate']*100:5.2f}%  "
                  f"steps={metrics['avg_steps_all']:5.1f}  "
                  f"ep_ms={metrics['avg_episode_ms']:5.2f}  "
                  f"step_μs={metrics['avg_step_us']:5.1f}  "
                  f"({train_time:.1f}s)")

    total_time = time.time() - t_start
    print(f"\n[compare] 全部完成, 总耗时 {total_time/60:.1f} 分钟")

    # ── 聚合: 每个策略算 mean ± std ──────────────────────
    summary = {}
    for name, seed_results in results.items():
        wr       = [r["win_rate"]               for r in seed_results.values()]
        cwr      = [r["conditional_win_rate"]   for r in seed_results.values()]
        fcd      = [r["first_click_deaths"]     for r in seed_results.values()]
        fcdr     = [r["first_click_death_rate"] for r in seed_results.values()]
        all_st   = [r["avg_steps_all"]          for r in seed_results.values()]
        sw       = [r["avg_steps_win"]          for r in seed_results.values() if r["avg_steps_win"] > 0]
        sl       = [r["avg_steps_loss"]         for r in seed_results.values()]
        cov      = [r["avg_coverage"]           for r in seed_results.values()]
        ep_ms    = [r["avg_episode_ms"]         for r in seed_results.values()]
        step_us  = [r["avg_step_us"]            for r in seed_results.values()]

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

    # ── 保存 JSON ─────────────────────────────────────
    with open(outdir / "results.json", "w") as f:
        json.dump({
            "per_seed": results,
            "summary":  summary,
        }, f, indent=2, ensure_ascii=False)

    # ── 生成可读表格 ──────────────────────────────────
    # 拆两张表, 免得一行超宽挤成一团
    lines = []
    lines.append(f"ε-策略对比实验   棋盘 {args.rows}x{args.cols} / {args.mines} 雷")
    lines.append(f"训练 {args.episodes} 局, 测试 {args.test_episodes} 局, seeds={args.seeds}")
    lines.append(f"其中 'Decay' + seed=42 对应 train.py 默认配置")
    lines.append("")

    # 表 1: 胜率 / 步数
    lines.append("【表 1】胜率 & 步数")
    h1 = f"{'Strategy':<20}{'Win Rate':>18}{'Steps(All)':>18}{'Steps(Win)':>18}{'Steps(Loss)':>18}{'Coverage':>15}"
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
        lines.append(f"{name:<20}{wr:>18}{sa:>18}{sw:>18}{sl:>18}{cov:>15}")
    lines.append("=" * len(h1))
    lines.append("")

    # 表 2: 测试时间效率
    lines.append("【表 2】测试时间 (ε=0 纯贪婪推理)")
    h2 = f"{'Strategy':<20}{'Episode (ms)':>20}{'Per-step (μs)':>20}"
    lines.append("=" * len(h2))
    lines.append(h2)
    lines.append("=" * len(h2))
    for name, s in summary.items():
        em = f"{s['avg_episode_ms_mean']:6.2f} ± {s['avg_episode_ms_std']:5.2f}"
        sm = f"{s['avg_step_us_mean']:6.1f} ± {s['avg_step_us_std']:5.1f}"
        lines.append(f"{name:<20}{em:>20}{sm:>20}")
    lines.append("=" * len(h2))
    lines.append("")

    # 表 3: 首步运气分离
    lines.append("【表 3】首步运气分离 (first-click death 不计入 skill)")
    h3 = f"{'Strategy':<20}{'Win Rate':>18}{'First-Click Death':>22}{'Conditional Win Rate':>25}"
    lines.append("=" * len(h3))
    lines.append(h3)
    lines.append("=" * len(h3))
    for name, s in summary.items():
        wr   = f"{s['win_rate_mean']*100:5.2f}% ± {s['win_rate_std']*100:4.2f}"
        fcd  = f"{s['first_click_death_rate_mean']*100:5.2f}% ± {s['first_click_death_rate_std']*100:4.2f}"
        cwr  = f"{s['conditional_win_rate_mean']*100:5.2f}% ± {s['conditional_win_rate_std']*100:4.2f}"
        lines.append(f"{name:<20}{wr:>18}{fcd:>22}{cwr:>25}")
    lines.append("=" * len(h3))

    table = "\n".join(lines)
    print("\n" + table)
    with open(outdir / "summary.txt", "w") as f:
        f.write(table + "\n")

    print(f"\n[compare] 所有产物已保存到: {outdir}")
    print(f"[compare] 包括: curves/, weights/, config.json, results.json, summary.txt")


if __name__ == "__main__":
    main()