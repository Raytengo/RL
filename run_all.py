#!/usr/bin/env python3
"""
run_all.py — Minesweeper RL 一键全流程脚本

用法:
    # 完整实验（全部对比，约 2~3 小时）
    python run_all.py

    # 快速验证（smoke test，约 5 分钟）
    python run_all.py --quick

    # 只跑训练+评估，跳过对比实验
    python run_all.py --no-compare

    # 指定已有 run 目录，跳过训练直接评估
    python run_all.py --run training_evaluation/runs/20260422_080515_9x9_m10_main

流程:
    Step 1  训练    SARSA(λ) agent（20000 局）
    Step 2  评估    贪婪策略，1000 局，输出 evaluate_report.md
    Step 3  基线    随机 agent 对比
    Step 4  λ 消融  λ ∈ {0.0, 0.4, 0.8, 1.0}
    Step 5  ε 消融  Fixed / Decay / Adaptive / Decay+Adaptive
    Step 6  算法    SARSA(λ) vs Q-Learning vs MonteCarlo
    Step 7  难度    4×4 / 6×6 / 9×9
    Step 8  汇总    生成 results/summary_report.md
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def banner(title: str):
    w = 70
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def run(cmd: list[str], label: str) -> bool:
    """运行子进程，实时转发输出，返回是否成功。"""
    print(f"\n[run] {label}")
    print(f"[cmd] {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"[ok]  {label}  ({elapsed:.0f}s)")
        return True
    else:
        print(f"[err] {label} 失败 (returncode={result.returncode}, {elapsed:.0f}s)")
        return False


def find_latest_run() -> Path | None:
    """自动找最新的训练 run 目录。"""
    runs_dir = ROOT / "training_evaluation" / "runs"
    if not runs_dir.exists():
        return None
    candidates = [
        p for p in runs_dir.iterdir()
        if p.is_dir()
        and (p / "config.json").exists()
        and (p / "final_model.npz").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def find_latest_eval(prefix: str) -> Path | None:
    """从 eval_results 找最新的某类实验目录。"""
    eval_dir = ROOT / "training_evaluation" / "eval_results"
    if not eval_dir.exists():
        return None
    candidates = [p for p in eval_dir.iterdir()
                  if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


# ─────────────────────────────────────────────────────────────────────────────
# 解析参数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Minesweeper RL — 一键全流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--quick", action="store_true",
        help="快速模式：训练 3000 局，对比 500 局，smoke test 用",
    )
    p.add_argument(
        "--run", type=str, default=None,
        help="指定已有训练 run 目录，跳过训练步骤",
    )
    p.add_argument(
        "--no-compare", action="store_true",
        help="跳过所有对比实验（Step 3~7），只跑训练+评估",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="主训练随机种子（默认 42）",
    )
    p.add_argument(
        "--episodes", type=int, default=None,
        help="训练局数（覆盖 quick 模式的默认值）",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 各步骤
# ─────────────────────────────────────────────────────────────────────────────

def step_train(args) -> Path | None:
    """Step 1: 训练。返回 run 目录路径，失败返回 None。"""
    banner("Step 1 / 7  —  训练 SARSA(λ) agent")

    if args.run:
        run_dir = Path(args.run).resolve()
        if not run_dir.exists():
            print(f"[err] 指定的 --run 目录不存在: {run_dir}")
            return None
        print(f"[skip] 使用已有 run 目录: {run_dir}")
        return run_dir

    episodes = args.episodes or (3000 if args.quick else 20000)
    cmd = [
        sys.executable, "-m", "training_evaluation.train",
        "--episodes", str(episodes),
        "--seed", str(args.seed),
        "--tag", "main",
        "--print_every", "500" if not args.quick else "200",
    ]
    ok = run(cmd, f"train ({episodes} episodes)")
    if not ok:
        return None

    run_dir = find_latest_run()
    if run_dir is None:
        print("[err] 训练结束后找不到 run 目录")
    return run_dir


def step_evaluate(run_dir: Path, args) -> bool:
    """Step 2: 评估。"""
    banner("Step 2 / 7  —  评估已训练模型（贪婪策略）")
    episodes = 500 if args.quick else 1000
    cmd = [
        sys.executable, "-m", "training_evaluation.evaluate",
        "--run", str(run_dir),
        "--episodes", str(episodes),
        "--seed", str(args.seed),
    ]
    return run(cmd, f"evaluate ({episodes} episodes, ε=0)")


def step_baseline(args) -> bool:
    """Step 3: 随机基线对比。"""
    banner("Step 3 / 7  —  随机基线对比")
    episodes = 500 if args.quick else 1000
    seeds = ["42"] if args.quick else ["42", "43", "44"]
    cmd = [
        sys.executable, "-m", "training_evaluation.compare_baseline",
        "--test-episodes", str(episodes),
        "--seeds", *seeds,
    ]
    return run(cmd, "compare_baseline")


def step_lambda(args) -> bool:
    """Step 4: λ 消融实验。"""
    banner("Step 4 / 7  —  λ 超参消融 (λ ∈ {0.0, 0.4, 0.8, 1.0})")
    episodes = 500 if args.quick else 20000
    test_ep = 200 if args.quick else 1000
    seeds = ["42"] if args.quick else ["42", "43", "44"]
    cmd = [
        sys.executable, "-m", "training_evaluation.compare_lambda",
        "--episodes", str(episodes),
        "--test-episodes", str(test_ep),
        "--seeds", *seeds,
    ]
    ok = run(cmd, "compare_lambda")
    if ok:
        # 画图
        d = find_latest_eval("lambda_compare")
        if d:
            run([sys.executable, "-m", "training_evaluation.plot_compare",
                 "--run", str(d)], "plot lambda curves")
    return ok


def step_epsilon(args) -> bool:
    """Step 5: ε 策略消融实验。"""
    banner("Step 5 / 7  —  ε 策略消融 (Fixed / Decay / Adaptive / Decay+Adaptive)")
    episodes = 500 if args.quick else 20000
    test_ep = 200 if args.quick else 1000
    seeds = ["42"] if args.quick else ["42", "43", "44"]
    cmd = [
        sys.executable, "-m", "training_evaluation.compare_epsilon",
        "--episodes", str(episodes),
        "--test-episodes", str(test_ep),
        "--seeds", *seeds,
    ]
    ok = run(cmd, "compare_epsilon")
    if ok:
        d = find_latest_eval("eps_compare")
        if d:
            run([sys.executable, "-m", "training_evaluation.plot_compare",
                 "--run", str(d)], "plot epsilon curves")
    return ok


def step_algorithms(args) -> bool:
    """Step 6: 算法对比 SARSA vs Q-Learning vs MonteCarlo。"""
    banner("Step 6 / 7  —  算法对比 (SARSA / Q-Learning / MonteCarlo)")
    episodes = 500 if args.quick else 20000
    test_ep = 200 if args.quick else 1000
    seeds = ["42"] if args.quick else ["42", "43", "44"]
    cmd = [
        sys.executable, "-m", "training_evaluation.compare_algorithms",
        "--episodes", str(episodes),
        "--test-episodes", str(test_ep),
        "--seeds", *seeds,
    ]
    ok = run(cmd, "compare_algorithms")
    if ok:
        d = find_latest_eval("algo_compare")
        if d:
            run([sys.executable, "-m", "training_evaluation.plot_compare",
                 "--run", str(d)], "plot algorithm curves")
    return ok


def step_difficulty(args) -> bool:
    """Step 7: 难度对比 4×4 / 6×6 / 9×9。"""
    banner("Step 7 / 7  —  难度对比 (4×4 / 6×6 / 9×9)")
    cmd = [sys.executable, "-m", "training_evaluation.compare_difficulty"]
    if args.quick:
        cmd += ["--episodes", "500"]
    ok = run(cmd, "compare_difficulty")
    if ok:
        d = find_latest_eval("difficulty_compare")
        if d:
            run([sys.executable, "-m", "training_evaluation.plot_compare",
                 "--run", str(d)], "plot difficulty curves")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 汇总报告
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(run_dir: Path | None, statuses: dict[str, bool], args):
    """生成 results/summary_report.md。"""
    RESULTS_DIR.mkdir(exist_ok=True)

    lines = [
        "# Minesweeper RL — 实验汇总报告",
        "",
        f"**运行时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**模式**: {'快速 (quick)' if args.quick else '完整 (full)'}",
        "",
        "## 步骤执行状态",
        "",
        "| 步骤 | 名称 | 状态 |",
        "|------|------|------|",
    ]
    step_names = {
        "train":       "Step 1  训练 SARSA(λ)",
        "evaluate":    "Step 2  模型评估",
        "baseline":    "Step 3  随机基线对比",
        "lambda":      "Step 4  λ 消融",
        "epsilon":     "Step 5  ε 策略消融",
        "algorithms":  "Step 6  算法对比",
        "difficulty":  "Step 7  难度对比",
    }
    for i, (key, name) in enumerate(step_names.items(), 1):
        if key not in statuses:
            status = "⏭ 跳过"
        elif statuses[key]:
            status = "✅ 完成"
        else:
            status = "❌ 失败"
        lines.append(f"| {i} | {name} | {status} |")

    lines += ["", "## 训练模型"]
    if run_dir:
        lines.append(f"- 路径: `{run_dir}`")
        report_md = run_dir / "training_report.md"
        eval_md = run_dir / "evaluate_report.md"
        if report_md.exists():
            lines += ["", "### 训练报告", ""]
            lines += report_md.read_text(encoding="utf-8").splitlines()
        if eval_md.exists():
            lines += ["", "### 评估报告", ""]
            lines += eval_md.read_text(encoding="utf-8").splitlines()
    else:
        lines.append("- 训练失败，无模型")

    lines += ["", "## 对比实验产物"]
    eval_root = ROOT / "training_evaluation" / "eval_results"
    if eval_root.exists():
        for d in sorted(eval_root.iterdir()):
            if not d.is_dir():
                continue
            summary_txt = d / "summary.txt"
            if summary_txt.exists():
                lines += ["", f"### {d.name}", "```"]
                lines += summary_txt.read_text(encoding="utf-8").splitlines()
                lines.append("```")

    report_path = RESULTS_DIR / "summary_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[summary] 汇总报告已写入: {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    total_t0 = time.time()
    statuses: dict[str, bool] = {}

    print("\n" + "█" * 70)
    print("  Minesweeper RL — 一键全流程")
    mode = "快速 (quick)" if args.quick else "完整 (full)"
    print(f"  模式: {mode}  |  seed={args.seed}")
    print("█" * 70)

    # Step 1: 训练
    run_dir = step_train(args)
    statuses["train"] = run_dir is not None

    # Step 2: 评估
    if run_dir is not None:
        ok = step_evaluate(run_dir, args)
        statuses["evaluate"] = ok
    else:
        print("[warn] 跳过评估（训练失败）")

    # Steps 3~7: 对比实验（可选）
    if args.no_compare:
        print("\n[skip] --no-compare 已设置，跳过所有对比实验")
    else:
        statuses["baseline"]   = step_baseline(args)
        statuses["lambda"]     = step_lambda(args)
        statuses["epsilon"]    = step_epsilon(args)
        statuses["algorithms"] = step_algorithms(args)
        statuses["difficulty"] = step_difficulty(args)

    # Step 8: 汇总
    banner("汇总  —  生成 results/summary_report.md")
    report = write_summary(run_dir, statuses, args)

    # 最终统计
    total_elapsed = time.time() - total_t0
    n_ok  = sum(v for v in statuses.values() if v)
    n_all = len(statuses)

    print("\n" + "─" * 70)
    print(f"  全部完成  {n_ok}/{n_all} 步成功  总耗时 {total_elapsed/60:.1f} 分钟")
    print(f"  汇总报告: {report}")
    print("─" * 70 + "\n")

    failed = [k for k, v in statuses.items() if not v]
    if failed:
        print(f"[warn] 失败步骤: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
