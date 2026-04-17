"""
training_evaluation/train.py

SARSA(λ) agent 的训练脚本。

用法:
    python3 training_evaluation/train.py                       # 默认配置
    python3 training_evaluation/train.py --episodes 20000      # 自定义训练局数
    python3 training_evaluation/train.py --rows 9 --cols 9 --mines 10 --seed 42

产物 (存到 --outdir 指定的目录, 默认 training_evaluation/runs/<run_name>/):
    - config.json        本次训练的完整配置
    - log.csv            每局的逐条记录 (episode, win, reward, steps, ...)
    - final_model.npz    训练结束时的权重
    - training_summary.png 训练过程图
    - training_report.md  训练结果总结
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

# 让脚本可以从项目根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment import MinesweeperEnv
from algorithm.agent import SARSALambdaAgent
from training_evaluation.plot_training import save_training_summary_plot


# ─────────────────────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # 环境
    p.add_argument("--rows",  type=int, default=9)
    p.add_argument("--cols",  type=int, default=9)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--safe-open-reward-per-cell", type=float, default=0.05)
    p.add_argument("--win-reward", type=float, default=10.0)
    p.add_argument("--lose-reward", type=float, default=-10.0)
    p.add_argument("--repeat-reward", type=float, default=-0.5)

    # 训练长度
    p.add_argument("--episodes", type=int, default=20000,
                   help="训练局数")

    # agent 超参
    p.add_argument("--alpha",   type=float, default=0.01)
    p.add_argument("--gamma",   type=float, default=0.99)
    p.add_argument("--lam",     type=float, default=0.8)
    p.add_argument("--eps_start", type=float, default=0.3)
    p.add_argument("--eps_end",   type=float, default=0.05)
    p.add_argument("--eps_decay", type=float, default=0.9995,
                   help="每局 epsilon *= eps_decay, 直到降到 eps_end")

    # 记录 / checkpoint
    p.add_argument("--log_every",  type=int, default=1,
                   help="每 N 局写一次 csv (1 = 每局都写)")
    p.add_argument("--print_every", type=int, default=500,
                   help="每 N 局打印一次滑动胜率")
    p.add_argument("--ckpt_every", type=int, default=0,
                   help="每 N 局存一次中途 checkpoint；0 表示不存")
    p.add_argument("--window", type=int, default=500,
                   help="滑动胜率的窗口大小")

    # 其他
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default=None,
                   help="输出目录, 默认 training_evaluation/runs/<timestamp>")
    p.add_argument("--tag", type=str, default="",
                   help="自定义 run 名后缀, 方便区分实验")
    p.add_argument("--plot-smooth-window", type=int, default=200,
                   help="训练完成后自动出图时使用的滑动平均窗口")

    return p.parse_args()


def configure_console_output():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# 单局训练 (一个 episode 的完整 SARSA(λ) 循环)
# ─────────────────────────────────────────────────────────────

def run_episode(env, agent):
    """
    跑一局, 返回这一局的统计字典。

    过程严格按照队友 agent 模块的约定:
      - 局开始调用 reset_trace()
      - 每步用 (s, a, r, s', a', done) 调用 update
      - done=True 时 s_next / a_next 传 None
    """
    state = env.reset()
    agent.reset_trace()

    valid = env.get_valid_actions()
    action = agent.select_action(state, valid)

    episode_reward = 0.0
    steps = 0

    while True:
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        steps += 1

        if done:
            agent.update(state, action, reward, None, None, done=True)
            break

        next_valid = env.get_valid_actions()
        next_action = agent.select_action(next_state, next_valid)
        agent.update(state, action, reward, next_state, next_action, done=False)

        state, action = next_state, next_action

    return {
        "win":         int(info.get("win", False)),
        "reward":      episode_reward,
        "steps":       steps,                      # agent 做了多少次决策
        "trial_count": next_state["trial_count"]   # 总共翻开了多少格 (含 BFS 展开)
            if not info.get("win") else state["trial_count"],
    }


def write_training_report(
    out_path: Path,
    window: int,
    final_win_rate: float,
    total_wins: int,
    rewards: list[float],
    steps_list: list[int],
    trial_counts: list[int],
    wins_list: list[int],
):
    episodes = len(wins_list)
    losses = episodes - total_wins
    win_rewards = [reward for reward, win in zip(rewards, wins_list) if win]
    loss_rewards = [reward for reward, win in zip(rewards, wins_list) if not win]
    win_steps = [steps for steps, win in zip(steps_list, wins_list) if win]
    loss_steps = [steps for steps, win in zip(steps_list, wins_list) if not win]
    win_trials = [count for count, win in zip(trial_counts, wins_list) if win]
    loss_trials = [count for count, win in zip(trial_counts, wins_list) if not win]

    def avg(values):
        return float(np.mean(values)) if values else 0.0

    lines = [
        "## Core Results",
        f"- Success count: {total_wins}",
        f"- Failure count: {losses}",
        f"- Overall win rate: {total_wins / episodes * 100:.3f}%",
        f"- Last {min(window, episodes)} episodes win rate: {final_win_rate * 100:.3f}%",
        f"- Average reward: {avg(rewards):.4f}",
        f"- Average steps: {avg(steps_list):.4f}",
        f"- Average opened cells: {avg(trial_counts):.4f}",
        "",
        "## Split Metrics",
        f"- Average reward on wins: {avg(win_rewards):.4f}",
        f"- Average reward on losses: {avg(loss_rewards):.4f}",
        f"- Average steps on wins: {avg(win_steps):.4f}",
        f"- Average steps on losses: {avg(loss_steps):.4f}",
        f"- Average opened cells on wins: {avg(win_trials):.4f}",
        f"- Average opened cells on losses: {avg(loss_trials):.4f}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# 主训练循环
# ─────────────────────────────────────────────────────────────

def main():
    configure_console_output()
    args = parse_args()

    # 固定随机种子, 保证可复现
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 准备输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.rows}x{args.cols}_m{args.mines}"
    if args.tag:
        run_name += f"_{args.tag}"

    outdir = Path(args.outdir) if args.outdir else \
             Path(__file__).resolve().parent / "runs" / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = outdir / "checkpoints"
    if args.ckpt_every > 0:
        checkpoints_dir.mkdir(exist_ok=True)

    # 保存配置
    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[train] run 目录: {outdir}")
    print(f"[train] 配置: {vars(args)}")

    # 初始化环境和 agent
    env = MinesweeperEnv(
        grid_size=(args.rows, args.cols),
        num_mines=args.mines,
        safe_open_reward_per_cell=args.safe_open_reward_per_cell,
        win_reward=args.win_reward,
        lose_reward=args.lose_reward,
        repeat_reward=args.repeat_reward,
    )
    agent = SARSALambdaAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        lam=args.lam,
        epsilon=args.eps_start,
    )

    # 打开 csv 日志
    log_file = open(outdir / "log.csv", "w", newline="")
    writer = csv.DictWriter(
        log_file,
        fieldnames=["episode", "win", "reward", "steps", "trial_count",
                    "epsilon", "w_norm"],
    )
    writer.writeheader()

    # 训练主循环
    recent_wins = []        # 用于滑动胜率打印
    start_time = time.time()
    rewards = []
    steps_list = []
    trial_counts = []
    wins_list = []

    for ep in range(1, args.episodes + 1):
        stats = run_episode(env, agent)
        rewards.append(stats["reward"])
        steps_list.append(stats["steps"])
        trial_counts.append(stats["trial_count"])
        wins_list.append(stats["win"])

        recent_wins.append(stats["win"])
        if len(recent_wins) > args.window:
            recent_wins.pop(0)

        # 写日志
        if ep % args.log_every == 0:
            writer.writerow({
                "episode":     ep,
                "win":         stats["win"],
                "reward":      f"{stats['reward']:.4f}",
                "steps":       stats["steps"],
                "trial_count": stats["trial_count"],
                "epsilon":     f"{agent.epsilon:.5f}",
                "w_norm":      f"{np.linalg.norm(agent.w):.4f}",
            })

        # 周期性打印进度
        if ep % args.print_every == 0:
            win_rate = sum(recent_wins) / len(recent_wins)
            elapsed = time.time() - start_time
            print(
                f"[ep {ep:>6d}] "
                f"win_rate(last {len(recent_wins)}) = {win_rate*100:5.2f}%  "
                f"eps = {agent.epsilon:.3f}  "
                f"‖w‖ = {np.linalg.norm(agent.w):.3f}  "
                f"elapsed = {elapsed:.1f}s"
            )

        # 周期性存 checkpoint
        if args.ckpt_every > 0 and ep % args.ckpt_every == 0:
            ckpt_path = checkpoints_dir / f"ep_{ep:06d}.npz"
            np.savez(ckpt_path, w=agent.w, episode=ep,
                     epsilon=agent.epsilon)

        # epsilon 衰减
        agent.epsilon = max(args.eps_end, agent.epsilon * args.eps_decay)

    log_file.close()

    # 最终模型
    np.savez(
        outdir / "final_model.npz",
        w=agent.w,
        episode=args.episodes,
        epsilon=agent.epsilon,
    )

    # 训练总结
    total_time = time.time() - start_time
    final_win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
    total_wins = sum(wins_list)
    print(f"\n[train] 训练完成, 总用时 {total_time:.1f}s")
    print(f"[train] 最后 {args.window} 局胜率: {final_win_rate*100:.2f}%")
    print(f"[train] 权重保存到: {outdir / 'final_model.npz'}")
    print(f"[train] 日志保存到: {outdir / 'log.csv'}")

    plot_path = save_training_summary_plot(
        run_dir=outdir.resolve(),
        out_path=(outdir / "training_summary.png").resolve(),
        smooth_window=args.plot_smooth_window,
    )
    print(f"[train] 训练图保存到: {plot_path}")

    report_path = (outdir / "training_report.md").resolve()
    write_training_report(
        out_path=report_path,
        window=args.window,
        final_win_rate=final_win_rate,
        total_wins=total_wins,
        rewards=rewards,
        steps_list=steps_list,
        trial_counts=trial_counts,
        wins_list=wins_list,
    )
    print(f"[train] 训练报告保存到: {report_path}")


if __name__ == "__main__":
    main()
    
