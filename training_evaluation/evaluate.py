"""
training_evaluation/evaluate.py

评估脚本：对训练好的模型和各种 baseline 进行全面评估。

用法:
    # 评估已训练的 SARSA(λ) 模型
    python3 training_evaluation/evaluate.py --model training_evaluation/runs/某次实验/final_model.npz

    # 只跑 baseline 对比 (不需要预训练模型)
    python3 training_evaluation/evaluate.py --skip-sarsa

    # 自定义测试局数和棋盘
    python3 training_evaluation/evaluate.py --model xxx.npz --test-episodes 2000 --rows 9 --cols 9 --mines 10

产物 (存到 --outdir, 默认 training_evaluation/eval_results/<timestamp>/):
    - eval_results.json     所有指标的完整结果
    - summary.txt           可读的对比表格
    - training_curve.csv    从训练 log 提取的曲线数据 (如果提供了 --train-log)

评估的方法:
    1. Random          纯随机选合法动作
    2. SARSA(λ)        加载训练好的模型, ε=0 纯贪婪
    3. Q-Learning      同样的特征, Q-Learning 更新规则, 现场训练
    4. Monte Carlo     同样的特征, MC 每局末尾更新, 现场训练

评估指标:
    - Win rate (胜率)
    - Average steps - win (胜局平均步数)
    - Average steps - loss (败局平均存活步数)
    - Coverage (平均翻开比例)

超参 ablation:
    - λ = 0, 0.4, 0.8, 1.0 的 SARSA 对比
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
from algorithm.features import extract_features, FEATURE_DIM


# ═══════════════════════════════════════════════════════════
# Agent 实现 (evaluate 自带, 不依赖队友的 agent.py)
# ═══════════════════════════════════════════════════════════

class BaseLinearAgent:
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

    def select_greedy(self, state, valid_actions):
        """纯贪婪, 评估时用"""
        return max(valid_actions, key=lambda a: self.Q(state, a))


class SARSALambdaAgent(BaseLinearAgent):
    """SARSA(λ), 和队友的实现逻辑一致"""

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


class QLearningAgent(BaseLinearAgent):
    """Q-Learning (off-policy), 用 max Q(s', a') 更新"""

    def update(self, s, a, r, s_next, valid_next, done):
        phi = extract_features(s, a)
        if done:
            target = r
        else:
            max_q = max(self.Q(s_next, a2) for a2 in valid_next)
            target = r + self.gamma * max_q
        delta = target - self.w @ phi
        self.w += self.alpha * delta * phi


class MonteCarloAgent(BaseLinearAgent):
    """Every-visit Monte Carlo, 一局结束后用累积 return 更新"""

    def __init__(self, alpha=0.01, gamma=0.99, epsilon=0.3):
        super().__init__(alpha, gamma, epsilon)
        self.episode_history = []

    def reset_episode(self):
        self.episode_history = []

    def record(self, state, action):
        self.episode_history.append((state, action))

    def update_episode(self, final_reward, step_rewards):
        """一局结束后, 从后往前算 return 并更新"""
        G = 0
        # step_rewards[i] 是第 i 步获得的 reward
        for t in range(len(self.episode_history) - 1, -1, -1):
            G = step_rewards[t] + self.gamma * G
            state, action = self.episode_history[t]
            phi = extract_features(state, action)
            delta = G - self.w @ phi
            self.w += self.alpha * delta * phi


# ═══════════════════════════════════════════════════════════
# 训练函数 (用于现场训练 Q-Learning 和 MC)
# ═══════════════════════════════════════════════════════════

def train_sarsa_lambda(env, episodes, alpha=0.01, gamma=0.99, lam=0.8,
                       eps_start=0.3, eps_end=0.05, eps_decay=0.9995):
    """训练 SARSA(λ), 返回训练好的 agent"""
    agent = SARSALambdaAgent(alpha=alpha, gamma=gamma, lam=lam, epsilon=eps_start)
    for ep in range(episodes):
        state = env.reset()
        agent.reset_trace()
        valid = env.get_valid_actions()
        action = agent.select_action(state, valid)
        while True:
            next_state, reward, done, info = env.step(action)
            if done:
                agent.update(state, action, reward, None, None, done=True)
                break
            next_valid = env.get_valid_actions()
            next_action = agent.select_action(next_state, next_valid)
            agent.update(state, action, reward, next_state, next_action, done=False)
            state, action = next_state, next_action
        agent.epsilon = max(eps_end, agent.epsilon * eps_decay)
    return agent


def train_qlearning(env, episodes, alpha=0.01, gamma=0.99,
                    eps_start=0.3, eps_end=0.05, eps_decay=0.9995):
    """训练 Q-Learning, 返回训练好的 agent"""
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=eps_start)
    for ep in range(episodes):
        state = env.reset()
        while True:
            valid = env.get_valid_actions()
            action = agent.select_action(state, valid)
            next_state, reward, done, info = env.step(action)
            if done:
                agent.update(state, action, reward, None, None, done=True)
                break
            next_valid = env.get_valid_actions()
            agent.update(state, action, reward, next_state, next_valid, done=False)
            state = next_state
        agent.epsilon = max(eps_end, agent.epsilon * eps_decay)
    return agent


def train_monte_carlo(env, episodes, alpha=0.01, gamma=0.99,
                      eps_start=0.3, eps_end=0.05, eps_decay=0.9995):
    """训练 Monte Carlo, 返回训练好的 agent"""
    agent = MonteCarloAgent(alpha=alpha, gamma=gamma, epsilon=eps_start)
    for ep in range(episodes):
        state = env.reset()
        agent.reset_episode()
        step_rewards = []
        while True:
            valid = env.get_valid_actions()
            action = agent.select_action(state, valid)
            agent.record(state, action)
            next_state, reward, done, info = env.step(action)
            step_rewards.append(reward)
            if done:
                agent.update_episode(reward, step_rewards)
                break
            state = next_state
        agent.epsilon = max(eps_end, agent.epsilon * eps_decay)
    return agent


# ═══════════════════════════════════════════════════════════
# 评估函数
# ═══════════════════════════════════════════════════════════

def evaluate_agent(env, agent, episodes, use_greedy=True):
    """
    评估一个 agent, 返回指标字典。
    agent=None 时用纯随机策略。
    """
    wins = 0
    steps_win = []
    steps_loss = []
    coverages = []
    total_safe = env.rows * env.cols - env.num_mines

    for _ in range(episodes):
        state = env.reset()
        steps = 0

        while not state["done"]:
            valid = env.get_valid_actions()
            if not valid:
                break
            if agent is None:
                action = random.choice(valid)
            elif use_greedy:
                action = agent.select_greedy(state, valid)
            else:
                action = agent.select_action(state, valid)
            state, reward, done, info = env.step(action)
            steps += 1

        won = info.get("win", False)
        if won:
            wins += 1
            steps_win.append(steps)
        else:
            steps_loss.append(steps)

        opened = state["trial_count"]
        coverages.append(opened / total_safe if total_safe > 0 else 0)

    return {
        "win_rate":          wins / episodes,
        "avg_steps_win":     np.mean(steps_win) if steps_win else 0,
        "avg_steps_loss":    np.mean(steps_loss) if steps_loss else 0,
        "avg_coverage":      np.mean(coverages),
        "total_wins":        wins,
        "total_episodes":    episodes,
    }


# ═══════════════════════════════════════════════════════════
# 输出格式化
# ═══════════════════════════════════════════════════════════

def format_results_table(all_results):
    """把结果格式化成可读表格"""
    lines = []
    header = f"{'Method':<25} {'Win Rate':>10} {'Steps(Win)':>12} {'Steps(Loss)':>12} {'Coverage':>10}"
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))

    for name, res in all_results.items():
        wr = f"{res['win_rate']*100:.2f}%"
        sw = f"{res['avg_steps_win']:.1f}" if res['avg_steps_win'] > 0 else "N/A"
        sl = f"{res['avg_steps_loss']:.1f}" if res['avg_steps_loss'] > 0 else "N/A"
        cov = f"{res['avg_coverage']*100:.1f}%"
        lines.append(f"{name:<25} {wr:>10} {sw:>12} {sl:>12} {cov:>10}")

    lines.append("=" * len(header))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None,
                   help="预训练 SARSA(λ) 模型路径 (.npz)")
    p.add_argument("--train-log", type=str, default=None,
                   help="训练 log.csv 路径, 用于生成训练曲线")
    p.add_argument("--skip-sarsa", action="store_true",
                   help="跳过预训练 SARSA(λ) 的评估")

    # 环境
    p.add_argument("--rows",  type=int, default=9)
    p.add_argument("--cols",  type=int, default=9)
    p.add_argument("--mines", type=int, default=10)

    # 评估 & 训练
    p.add_argument("--test-episodes",  type=int, default=1000,
                   help="每个方法的测试局数")
    p.add_argument("--train-episodes", type=int, default=20000,
                   help="现场训练 Q-Learning / MC / λ-ablation 的局数")

    # 超参 (用于现场训练的方法)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)

    # 输出
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else \
             Path(__file__).resolve().parent / "eval_results" / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    env = MinesweeperEnv(grid_size=(args.rows, args.cols), num_mines=args.mines)
    all_results = {}

    print(f"[eval] 棋盘: {args.rows}x{args.cols}, {args.mines} 雷")
    print(f"[eval] 测试局数: {args.test_episodes}")
    print(f"[eval] 现场训练局数: {args.train_episodes}")
    print()

    # ── 1. Random Baseline ──────────────────────────────
    print("[eval] 1/6  评估 Random baseline ...")
    t0 = time.time()
    all_results["Random"] = evaluate_agent(env, None, args.test_episodes)
    print(f"       完成, 胜率 {all_results['Random']['win_rate']*100:.2f}%, "
          f"用时 {time.time()-t0:.1f}s")

    # ── 2. 预训练 SARSA(λ) ──────────────────────────────
    if not args.skip_sarsa and args.model:
        print(f"[eval] 2/6  加载预训练模型: {args.model}")
        data = np.load(args.model)
        sarsa_agent = SARSALambdaAgent()
        sarsa_agent.w = data["w"]
        t0 = time.time()
        all_results["SARSA(λ) pretrained"] = evaluate_agent(
            env, sarsa_agent, args.test_episodes)
        print(f"       完成, 胜率 {all_results['SARSA(λ) pretrained']['win_rate']*100:.2f}%, "
              f"用时 {time.time()-t0:.1f}s")
    else:
        print("[eval] 2/6  跳过预训练 SARSA(λ)")

    # ── 3. Q-Learning (现场训练) ─────────────────────────
    print(f"[eval] 3/6  训练 Q-Learning ({args.train_episodes} 局) ...")
    t0 = time.time()
    ql_agent = train_qlearning(env, args.train_episodes,
                               alpha=args.alpha, gamma=args.gamma)
    train_time = time.time() - t0
    print(f"       训练完成, 用时 {train_time:.1f}s, 评估中 ...")
    t0 = time.time()
    all_results["Q-Learning"] = evaluate_agent(env, ql_agent, args.test_episodes)
    print(f"       完成, 胜率 {all_results['Q-Learning']['win_rate']*100:.2f}%, "
          f"用时 {time.time()-t0:.1f}s")

    # ── 4. Monte Carlo (现场训练) ────────────────────────
    print(f"[eval] 4/6  训练 Monte Carlo ({args.train_episodes} 局) ...")
    t0 = time.time()
    mc_agent = train_monte_carlo(env, args.train_episodes,
                                 alpha=args.alpha, gamma=args.gamma)
    train_time = time.time() - t0
    print(f"       训练完成, 用时 {train_time:.1f}s, 评估中 ...")
    t0 = time.time()
    all_results["Monte Carlo"] = evaluate_agent(env, mc_agent, args.test_episodes)
    print(f"       完成, 胜率 {all_results['Monte Carlo']['win_rate']*100:.2f}%, "
          f"用时 {time.time()-t0:.1f}s")

    # ── 5. λ ablation ───────────────────────────────────
    print(f"[eval] 5/6  λ ablation (各 {args.train_episodes} 局训练) ...")
    lambda_values = [0.0, 0.4, 0.8, 1.0]
    for lam_val in lambda_values:
        label = f"SARSA(λ={lam_val})"
        print(f"       训练 {label} ...")
        t0 = time.time()
        agent = train_sarsa_lambda(env, args.train_episodes,
                                   alpha=args.alpha, gamma=args.gamma,
                                   lam=lam_val)
        train_time = time.time() - t0
        all_results[label] = evaluate_agent(env, agent, args.test_episodes)
        wr = all_results[label]['win_rate'] * 100
        print(f"       {label}: 胜率 {wr:.2f}%, 训练 {train_time:.1f}s")

    # ── 6. 汇总输出 ─────────────────────────────────────
    print(f"\n[eval] 6/6  汇总结果\n")
    table = format_results_table(all_results)
    print(table)

    # 保存 JSON
    json_results = {}
    for name, res in all_results.items():
        json_results[name] = {k: float(v) for k, v in res.items()}

    with open(outdir / "eval_results.json", "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    # 保存可读摘要
    with open(outdir / "summary.txt", "w") as f:
        f.write(f"评估配置: {args.rows}x{args.cols}, {args.mines} 雷\n")
        f.write(f"测试局数: {args.test_episodes}\n")
        f.write(f"训练局数: {args.train_episodes}\n")
        f.write(f"seed: {args.seed}\n\n")
        f.write(table)
        f.write("\n")

    print(f"\n[eval] 结果已保存到: {outdir}")
    print(f"       - eval_results.json")
    print(f"       - summary.txt")


if __name__ == "__main__":
    main()
