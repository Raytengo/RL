"""
training_evaluation/compare_algorithms.py

算法对比实验: SARSA(λ) vs Q-Learning vs Monte Carlo
---------------------------------------------------
变量控制:
  - 共享的 features.py (10 维) 和 MinesweeperEnv (不改)
  - 共享超参: alpha=0.01, gamma=0.99, episodes=20000
  - 共享 ε 策略: Decay (0.3 → 0.05, decay=0.9995), 和 train.py 默认一致
  - 共享 seed 集合 (默认 42, 43, 44, 对每个算法都跑一遍取均值)
  - 唯一变量: 算法类型 (SARSA / Q-Learning / Monte Carlo)

三种算法的核心差异:
  SARSA(λ):   on-policy, bootstrapping, 带 eligibility trace (λ=0.8)
  Q-Learning: off-policy, bootstrapping, 无 trace
  Monte Carlo: on-policy, 不 bootstrapping, 整局结束后用累积 return 更新

用法:
    # 默认 Decay 策略 (和 train.py 一致, 80 分钟)
    python3 training_evaluation/compare_algorithms.py

    # 使用 Decay+Adaptive 策略 (你们 ε 实验的最好组合)
    python training_evaluation/compare_algorithms.py --eps-strategy decay_adaptive

    # 快速 smoke test
    python3 training_evaluation/compare_algorithms.py --episodes 500 --seeds 42 --test-episodes 200

产物: training_evaluation/eval_results/algo_compare_<eps_strategy>_<timestamp>/
    - config.json      本次实验配置
    - results.json     每个 (算法, seed) 的完整指标
    - summary.txt      可读的对比表
    - curves/*.csv     滑动胜率曲线
    - weights/*.npz    训练好的 agent 权重
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
from algorithm.features import extract_features, FEATURE_DIM


# ═══════════════════════════════════════════════════════════
# Agent 实现 (全部共享 features.extract_features)
# ═══════════════════════════════════════════════════════════

class BaseAgent:
    """线性函数近似的基类, 共享特征和选动作逻辑"""

    def __init__(self, alpha=0.01, gamma=0.99, epsilon=0.3):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.w = np.zeros(FEATURE_DIM)

    def Q(self, state, action):
        return float(self.w @ extract_features(state, action))

    def select_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        return max(valid_actions, key=lambda a: self.Q(state, a))


class SARSALambdaAgent(BaseAgent):
    """SARSA(λ): on-policy + trace. 和队友 agent.py 的实现逻辑一致."""

    def __init__(self, alpha=0.01, gamma=0.99, lam=0.8, epsilon=0.3):
        super().__init__(alpha, gamma, epsilon)
        self.lam = lam
        self.e = np.zeros(FEATURE_DIM)

    def reset_trace(self):
        self.e = np.zeros(FEATURE_DIM)

    def update(self, s, a, r, s_next, a_next, done):
        phi = extract_features(s, a)
        if done:
            delta = r - self.w @ phi
        else:
            delta = r + self.gamma * self.Q(s_next, a_next) - self.w @ phi
        self.e = self.gamma * self.lam * self.e + phi
        self.w += self.alpha * delta * self.e


class QLearningAgent(BaseAgent):
    """Q-Learning: off-policy. 用 max_a' Q(s', a') 作为 target, 不带 trace."""

    def update(self, s, a, r, s_next, valid_next, done):
        phi = extract_features(s, a)
        if done:
            target = r
        else:
            max_q = max(self.Q(s_next, a2) for a2 in valid_next)
            target = r + self.gamma * max_q
        delta = target - self.w @ phi
        self.w += self.alpha * delta * phi


class MonteCarloAgent(BaseAgent):
    """
    Every-visit Monte Carlo (first-visit-style 也可以但这里用 every-visit 更简单).
    每局结束后, 从后往前算 G_t, 对每个 (s, a) 做 w += α * (G - w·φ) * φ.
    不 bootstrapping, 整局结束才更新.
    """

    def __init__(self, alpha=0.01, gamma=0.99, epsilon=0.3):
        super().__init__(alpha, gamma, epsilon)
        self._trajectory = []       # [(s, a, r), ...] 这一局的轨迹

    def reset_trace(self):
        """MC 没有 trace, 但为了统一接口, 用来清空本局轨迹"""
        self._trajectory = []

    def record_step(self, s, a, r):
        """每一步调用, 把 (s, a, r) 追加到本局轨迹"""
        self._trajectory.append((s, a, r))

    def update_episode(self):
        """局结束后调用, 用累积 return 更新权重"""
        G = 0.0
        # 从后往前
        for s, a, r in reversed(self._trajectory):
            G = r + self.gamma * G
            phi = extract_features(s, a)
            delta = G - self.w @ phi
            self.w += self.alpha * delta * phi


# ═══════════════════════════════════════════════════════════
# ε-策略 (和 compare_epsilon.py 里的 Decay 一致)
# ═══════════════════════════════════════════════════════════

class DecayEpsilonStrategy:
    """
    跨局线性衰减, 局内不变. 和 train.py 默认配置一致.
    eps_base -> max(eps_end, eps_base * eps_decay) 每局
    """
    def __init__(self, eps_start=0.3, eps_end=0.05, eps_decay=0.9995):
        self.eps_base = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def get_epsilon(self, state=None, env=None):
        return self.eps_base

    def on_episode_end(self):
        self.eps_base = max(self.eps_end, self.eps_base * self.eps_decay)


class DecayAdaptiveEpsilonStrategy:
    """
    Decay + Adaptive: 跨局衰减 + 局内按 opened_ratio 自适应.
    ε_local = ε_base × (1 - opened_ratio)²   (方案 B, 幂函数)
    和 compare_epsilon.py 里的 Decay+Adaptive 一致.
    """
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


def build_strategy(name: str):
    """根据名字构造 ε 策略实例."""
    if name == "decay":
        return DecayEpsilonStrategy()
    elif name == "decay_adaptive":
        return DecayAdaptiveEpsilonStrategy()
    else:
        raise ValueError(f"未知的 ε 策略: {name}. 可选: decay, decay_adaptive")


# ═══════════════════════════════════════════════════════════
# 训练一个 run (给定算法 + seed)
# ═══════════════════════════════════════════════════════════

def train_sarsa_run(seed, args):
    random.seed(seed)
    np.random.seed(seed)

    env = MinesweeperEnv(grid_size=(args.rows, args.cols), num_mines=args.mines)
    strategy = build_strategy(args.eps_strategy)
    agent = SARSALambdaAgent(alpha=args.alpha, gamma=args.gamma, lam=args.lam,
                             epsilon=strategy.eps_base)
    win_history = []

    for ep in range(args.episodes):
        state = env.reset()
        agent.reset_trace()
        valid = env.get_valid_actions()
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


def train_qlearning_run(seed, args):
    random.seed(seed)
    np.random.seed(seed)

    env = MinesweeperEnv(grid_size=(args.rows, args.cols), num_mines=args.mines)
    strategy = build_strategy(args.eps_strategy)
    agent = QLearningAgent(alpha=args.alpha, gamma=args.gamma, epsilon=strategy.eps_base)
    win_history = []

    for ep in range(args.episodes):
        state = env.reset()
        while True:
            valid = env.get_valid_actions()
            agent.epsilon = strategy.get_epsilon(state, env)
            action = agent.select_action(state, valid)
            next_state, reward, done, info = env.step(action)
            if done:
                agent.update(state, action, reward, None, None, done=True)
                break
            next_valid = env.get_valid_actions()
            agent.update(state, action, reward, next_state, next_valid, done=False)
            state = next_state

        win_history.append(1 if info.get("win") else 0)
        strategy.on_episode_end()

    return agent, win_history


def train_mc_run(seed, args):
    random.seed(seed)
    np.random.seed(seed)

    env = MinesweeperEnv(grid_size=(args.rows, args.cols), num_mines=args.mines)
    strategy = build_strategy(args.eps_strategy)
    agent = MonteCarloAgent(alpha=args.alpha, gamma=args.gamma, epsilon=strategy.eps_base)
    win_history = []

    for ep in range(args.episodes):
        state = env.reset()
        agent.reset_trace()
        while True:
            valid = env.get_valid_actions()
            agent.epsilon = strategy.get_epsilon(state, env)
            action = agent.select_action(state, valid)
            next_state, reward, done, info = env.step(action)
            agent.record_step(state, action, reward)
            if done:
                agent.update_episode()
                break
            state = next_state

        win_history.append(1 if info.get("win") else 0)
        strategy.on_episode_end()

    return agent, win_history


TRAIN_FNS = {
    "SARSA":       train_sarsa_run,
    "Q-Learning":  train_qlearning_run,
    "MonteCarlo":  train_mc_run,
}


# ═══════════════════════════════════════════════════════════
# 评估 (ε=0 纯贪婪)
# ═══════════════════════════════════════════════════════════

def evaluate(agent, args, seed):
    random.seed(seed + 10000)
    np.random.seed(seed + 10000)

    env = MinesweeperEnv(grid_size=(args.rows, args.cols), num_mines=args.mines)
    agent.epsilon = 0.0

    wins = 0
    first_click_deaths = 0   # 第一步就踩雷的局数 (纯运气, 不怪 agent)
    steps_all, steps_win, steps_loss, coverages = [], [], [], []
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
            # 第一步就挂 = 开局无任何信息, 纯运气
            if steps == 1:
                first_click_deaths += 1
        coverages.append(state["trial_count"] / total_safe if total_safe > 0 else 0)

    # 条件胜率: 排除首步即死的局后的胜率 ("纯 skill 胜率")
    non_first_click_deaths = args.test_episodes - first_click_deaths
    conditional_win_rate = wins / non_first_click_deaths if non_first_click_deaths > 0 else 0.0

    return {
        "win_rate":             wins / args.test_episodes,
        "conditional_win_rate": conditional_win_rate,
        "first_click_deaths":   first_click_deaths,
        "first_click_death_rate": first_click_deaths / args.test_episodes,
        "avg_steps_all":        float(np.mean(steps_all)),
        "avg_steps_win":        float(np.mean(steps_win))  if steps_win  else 0.0,
        "avg_steps_loss":       float(np.mean(steps_loss)) if steps_loss else 0.0,
        "avg_coverage":         float(np.mean(coverages)),
        "avg_episode_ms":       float(np.mean(episode_times_ms)),
        "avg_step_us":          float(np.mean(step_times_us)) if step_times_us else 0.0,
        "total_wins":           wins,
    }


# ═══════════════════════════════════════════════════════════
# 滑动胜率 (训练曲线用)
# ═══════════════════════════════════════════════════════════

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
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--alpha",    type=float, default=0.01)
    p.add_argument("--gamma",    type=float, default=0.99)
    p.add_argument("--lam",      type=float, default=0.8, help="仅用于 SARSA")
    p.add_argument("--test-episodes", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--curve-window", type=int, default=500)
    p.add_argument("--algorithms", type=str, nargs="+",
                   default=["SARSA", "Q-Learning", "MonteCarlo"],
                   help="要跑的算法, 默认全部")
    p.add_argument("--eps-strategy", type=str, default="decay",
                   choices=["decay", "decay_adaptive"],
                   help="ε 策略: decay (和 train.py 默认一致) 或 decay_adaptive "
                        "(Decay 基础上再加局内 adaptive)")
    return p.parse_args()


def main():
    args = parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 把 ε 策略写进目录名, 避免两次 run 撞在一起
    eps_tag = args.eps_strategy   # "decay" or "decay_adaptive"
    outdir = Path(args.outdir) if args.outdir else \
             Path(__file__).resolve().parent / "eval_results" / f"algo_compare_{eps_tag}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "curves").mkdir(exist_ok=True)
    (outdir / "weights").mkdir(exist_ok=True)

    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    eps_label = "Decay (0.3→0.05, decay=0.9995)" if args.eps_strategy == "decay" \
                else "Decay+Adaptive (Decay × (1-opened_ratio)²)"
    print(f"[compare] 输出目录: {outdir}")
    print(f"[compare] 棋盘: {args.rows}x{args.cols}, {args.mines} 雷")
    print(f"[compare] 训练 {args.episodes} 局, 评估 {args.test_episodes} 局")
    print(f"[compare] Seeds: {args.seeds}")
    print(f"[compare] 算法: {args.algorithms}")
    print(f"[compare] ε 策略: {eps_label}\n")

    results = {name: {} for name in args.algorithms}
    total_runs = len(args.algorithms) * len(args.seeds)
    run_idx = 0
    t_start = time.time()

    for algo in args.algorithms:
        if algo not in TRAIN_FNS:
            print(f"[err] 未知算法: {algo}")
            continue
        train_fn = TRAIN_FNS[algo]

        for seed in args.seeds:
            run_idx += 1
            t0 = time.time()
            print(f"[{run_idx:2d}/{total_runs}] {algo:<12s} seed={seed} ... ",
                  end="", flush=True)

            agent, win_history = train_fn(seed, args)
            train_time = time.time() - t0

            metrics = evaluate(agent, args, seed)
            metrics["train_time_sec"] = train_time
            results[algo][seed] = metrics

            curve = sliding_win_rate(win_history, window=args.curve_window)
            curve_path = outdir / "curves" / f"{algo}_seed{seed}.csv"
            with open(curve_path, "w", newline="") as cf:
                w = csv.writer(cf)
                w.writerow(["episode", "win", "sliding_win_rate"])
                for i, (win, rate) in enumerate(zip(win_history, curve), 1):
                    w.writerow([i, win, f"{rate:.4f}"])

            # 保存 agent 权重
            weight_path = outdir / "weights" / f"{algo}_seed{seed}.npz"
            np.savez(weight_path,
                     w=agent.w,
                     algorithm=algo,
                     seed=seed,
                     episodes=args.episodes)

            print(f"win_rate={metrics['win_rate']*100:5.2f}%  "
                  f"steps={metrics['avg_steps_all']:5.1f}  "
                  f"ep_ms={metrics['avg_episode_ms']:5.2f}  "
                  f"step_μs={metrics['avg_step_us']:5.1f}  "
                  f"({train_time:.1f}s)")

    total_time = time.time() - t_start
    print(f"\n[compare] 全部完成, 总耗时 {total_time/60:.1f} 分钟")

    # 聚合 mean ± std
    summary = {}
    for algo, seed_results in results.items():
        if not seed_results:
            continue
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

        summary[algo] = {
            "win_rate_mean":       float(np.mean(wr)),
            "win_rate_std":        float(np.std(wr)),
            "conditional_win_rate_mean": float(np.mean(cwr)),
            "conditional_win_rate_std":  float(np.std(cwr)),
            "first_click_deaths_mean":   float(np.mean(fcd)),
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

    # 可读表格 (分两张, 免得一行过宽)
    lines = []
    lines.append(f"算法对比实验   棋盘 {args.rows}x{args.cols} / {args.mines} 雷")
    lines.append(f"训练 {args.episodes} 局, 测试 {args.test_episodes} 局, seeds={args.seeds}")
    lines.append(f"ε 策略: {eps_label}")
    lines.append(f"共享: 10 维特征, α={args.alpha}, γ={args.gamma}; SARSA 额外用 λ={args.lam}")
    lines.append("")

    lines.append("【表 1】胜率 & 步数")
    h1 = f"{'Algorithm':<14}{'Win Rate':>18}{'Steps(All)':>18}{'Steps(Win)':>18}{'Steps(Loss)':>18}{'Coverage':>15}"
    lines.append("=" * len(h1))
    lines.append(h1)
    lines.append("=" * len(h1))
    for algo, s in summary.items():
        wr = f"{s['win_rate_mean']*100:5.2f}% ± {s['win_rate_std']*100:4.2f}"
        sa = f"{s['avg_steps_all_mean']:5.1f} ± {s['avg_steps_all_std']:4.1f}"
        sw = f"{s['avg_steps_win_mean']:5.1f} ± {s['avg_steps_win_std']:4.1f}" \
            if s['avg_steps_win_mean'] > 0 else "N/A"
        sl = f"{s['avg_steps_loss_mean']:5.1f} ± {s['avg_steps_loss_std']:4.1f}"
        cov = f"{s['avg_coverage_mean']*100:5.1f}% ± {s['avg_coverage_std']*100:4.1f}"
        lines.append(f"{algo:<14}{wr:>18}{sa:>18}{sw:>18}{sl:>18}{cov:>15}")
    lines.append("=" * len(h1))
    lines.append("")

    lines.append("【表 2】测试时间 (ε=0 纯贪婪推理)")
    h2 = f"{'Algorithm':<14}{'Episode (ms)':>20}{'Per-step (μs)':>20}"
    lines.append("=" * len(h2))
    lines.append(h2)
    lines.append("=" * len(h2))
    for algo, s in summary.items():
        em = f"{s['avg_episode_ms_mean']:6.2f} ± {s['avg_episode_ms_std']:5.2f}"
        sm = f"{s['avg_step_us_mean']:6.1f} ± {s['avg_step_us_std']:5.1f}"
        lines.append(f"{algo:<14}{em:>20}{sm:>20}")
    lines.append("=" * len(h2))
    lines.append("")

    lines.append("【表 3】首步运气分离 (first-click death 不计入 skill)")
    h3 = f"{'Algorithm':<14}{'Win Rate':>18}{'First-Click Death':>22}{'Conditional Win Rate':>25}"
    lines.append("=" * len(h3))
    lines.append(h3)
    lines.append("=" * len(h3))
    for algo, s in summary.items():
        wr   = f"{s['win_rate_mean']*100:5.2f}% ± {s['win_rate_std']*100:4.2f}"
        fcd  = f"{s['first_click_death_rate_mean']*100:5.2f}% ± {s['first_click_death_rate_std']*100:4.2f}"
        cwr  = f"{s['conditional_win_rate_mean']*100:5.2f}% ± {s['conditional_win_rate_std']*100:4.2f}"
        lines.append(f"{algo:<14}{wr:>18}{fcd:>22}{cwr:>25}")
    lines.append("=" * len(h3))

    table = "\n".join(lines)
    print("\n" + table)
    with open(outdir / "summary.txt", "w") as f:
        f.write(table + "\n")

    print(f"\n[compare] 所有产物已保存到: {outdir}")
    print(f"[compare] 包括: curves/, weights/, config.json, results.json, summary.txt")


if __name__ == "__main__":
    main()