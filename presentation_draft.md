# Minesweeper RL: Learning to Sweep Mines
### Group Presentation Draft

---

## Slide 1 — Title

**Reinforcement Learning for Minesweeper**
*Implementing and Optimizing SARSA(λ), Q-Learning, and Monte Carlo*

> **Goal:** Implement the classic RL algorithms taught in class and push the win rate on Minesweeper as high as possible through principled iterative improvements.

---

## Slide 2 — Game Introduction: Why Minesweeper?

**What is Minesweeper?**
- A 9×9 grid with 10 hidden mines
- Each turn: choose an unrevealed cell to open
  - **Safe cell** → reveals a number (0–8) counting adjacent mines; if 0, auto-expands recursively (BFS)
  - **Mine** → game over
- **Win condition:** reveal every non-mine cell

**Why is it interesting for RL?**
- **Partial observability:** the mine layout is hidden; only revealed numbers give clues
- **Sequential decisions:** each choice changes the information state
- **Sparse yet structured rewards:** winning requires many correct steps; one mistake ends everything
- **Mix of logic and luck:** ~12% of games are decided entirely by the first random click

---

## Slide 3 — Project Pipeline Overview

Our development followed a five-phase pipeline:

```
[ Environment ]  →  [ Algorithms ]  →  [ Observe & Debug ]  →  [ Optimize ]  →  [ SOTA ]
   Env + API         SARSA/MC/QL          Replay watch           3 fixes          Results
   Rewards           Features (7-dim)     Problem analysis       Features×10
                     Decay ε                                     Adaptive ε
```

Key principle: **build, observe, diagnose, fix** — not tune blindly.

---

## Slide 4 — Environment Construction

**State representation** — a visible grid where each cell is one of:

| Value | Meaning |
|-------|---------|
| −1 | Hidden (not yet revealed) |
| 0–8 | Revealed: number of adjacent mines |
| 9 | Mine (revealed after explosion) |

**Standard RL interface:**
```python
state  = env.reset()                       # start a new game
valid  = env.get_valid_actions()           # all unrevealed cells: [(r,c), ...]
state, reward, done, info = env.step((r,c))  # take action
```

**BFS cascade:** clicking a "0" cell auto-reveals all connected zero-cells — exactly matching real Minesweeper rules.

**Two play modes built:**
- CLI mode — human types coordinates; useful for debugging
- GUI mode (Tkinter) — visual board for replay observation and demo

---

## Slide 5 — Reward Design

Carefully shaped to balance dense feedback with terminal dominance:

| Event | Reward |
|-------|--------|
| Safe cell opened (per cell) | +0.05 |
| Win (all safe cells revealed) | +10.0 |
| Hit a mine | −10.0 |
| Click an already-revealed cell | −0.5 |

**Design rationale:**
- Small per-cell reward (+0.05) provides a dense learning signal — the agent isn't flying blind
- Large terminal rewards (±10) ensure the agent's primary goal stays winning, not just opening cells
- The repeat-click penalty discourages degenerate loops

---

## Slide 6 — Algorithm Implementation

All three algorithms share **linear function approximation**:

> **Q(s, a) = w · φ(s, a)**

where φ(s, a) is a hand-crafted feature vector and w is learned.

### SARSA(λ) — On-Policy with Eligibility Traces
```
δ  = r + γ·Q(s', a') − Q(s, a)    # TD error
e  ← γλe + φ(s, a)                # eligibility trace
w  ← w + α·δ·e                    # weight update
```
Parameters: α=0.01, γ=0.99, λ=0.8

### Q-Learning — Off-Policy Bootstrapping
```
δ = r + γ·max_{a'} Q(s', a') − Q(s, a)
w ← w + α·δ·φ(s, a)
```

### Monte Carlo — Full-Episode Returns
Update w using the actual cumulative return G_t after each complete episode. No bootstrapping.

All three use ε-greedy action selection with **action masking** — the agent can only select unrevealed cells, eliminating invalid moves entirely.

---

## Slide 7 — Feature Engineering (Initial: 7-Dimensional)

Each feature encodes something the agent "should" care about for cell (r, c):

| Feature | Description |
|---------|-------------|
| f1 | Hidden neighbor ratio: `#hidden_neighbors / 8` |
| f2 | Numbered neighbor sum: `Σ neighbor_values / 64` |
| f3 | Danger estimate: `f2_sum / #hidden / 8` |
| f4 | Isolation flag: 1 if no numbered neighbors (cell is in unknown territory) |
| f5 | Global hidden ratio: fraction of board still hidden |
| f6 | Known mine ratio: fraction of board showing exploded mines |
| f7 | Bias: fixed at 1.0 (intercept term) |

**Baseline Decay ε Strategy:**
```
ε ← max(0.05, ε × 0.9995)    per episode
```
ε decays from 0.30 → 0.05 over 20,000 training episodes (explore early, exploit later).

---

## Slide 8 — Initial Results: Something Feels Wrong

After training 20,000 episodes with the three algorithms + decay ε:

| Algorithm | Training Win Rate (last 500 ep.) | Greedy Eval (1,000 ep.) |
|-----------|----------------------------------|-------------------------|
| SARSA(λ) | ~49% | 66.5% |
| Q-Learning | ~49% | 69.5% |
| Monte Carlo | ~35% | 55.6% (**high variance**) |
| **Random baseline** | — | **0.0%** |

The models outperform random significantly — but the **gap between training and eval** and Monte Carlo's **instability** prompted us to look closer.

> "Why does greedy eval score 15–20 points higher than training? What is ε-exploration actually costing us?"

---

## Slide 9 — Watching the Replay: Spotting the Problems

We used our GUI replay tool to watch the trained agent play in real time.

**What we observed:**

1. **Some games end on move 1 — before the agent does anything.**
   The very first action must be random (zero board information is available). Roughly 1 in 8 games ends right there.

2. **Late-game random clicking, despite knowing the answer.**
   When most of the board is revealed and logic can clearly identify which cells are safe, the agent still clicks randomly 5% of the time (ε=0.05). On a near-complete board this is costly.

Both problems share a root cause: **ε is a blunt, global instrument** — it doesn't respond to what the game state actually tells us.

---

## Slide 10 — Problem 1: First-Click Death

**The issue:**
- On the first move, all cells look identical — no information distinguishes safe from dangerous
- The agent must act randomly; mine probability = 10/81 ≈ **12.3%**
- This is **irreducible** — no algorithm can learn its way out of this

**Our fix — Conditional Win Rate:**
```
Conditional Win Rate = Wins / (Episodes − First-Click-Deaths)
```

This metric measures **agent skill**, not luck:

| Metric | SARSA(λ) Decay+Adaptive |
|--------|------------------------|
| Overall win rate | 69.43% |
| First-click death rate | 12.63% |
| **Conditional win rate** | **79.48%** |

Removing first-click noise gives a cleaner picture of what the RL agent actually learned.

---

## Slide 11 — Problem 2 & Fix: Adaptive In-Episode ε

**The issue:**
Global decay ε sets the same exploration rate for move 1 (board blank, exploration makes sense) and move 50 (board nearly solved, exploration is wasteful).

**Our solution — Adaptive In-Episode ε:**

```python
opened_ratio = state["trial_count"] / (rows × cols)
ε_local = ε_base × (1 − opened_ratio)²
```

Interpretation:
- Early game: `opened_ratio ≈ 0` → `ε_local ≈ ε_base` (full exploration)
- Late game: `opened_ratio → 1` → `ε_local → 0` (pure exploitation)

The **quadratic** schedule makes the transition smooth and front-loads exploration where it helps most.

| Strategy | Win Rate | Std Dev |
|----------|----------|---------|
| Fixed ε | 56.33% | ±6.96% |
| Decay only | 66.47% | ±0.80% |
| Adaptive only | 67.27% | ±0.17% |

Adaptive alone reduces **variance by 5×** compared to Fixed — the agent is now consistently using what it knows.

---

## Slide 12 — Combining Decay + Adaptive = SOTA

**Two orthogonal improvements, composable:**

```
# After each episode (cross-episode learning):
ε_base ← max(0.05, ε_base × 0.9995)

# At each step within an episode (within-game adaptation):
ε_local = ε_base × (1 − opened_ratio)²
```

- **Decay** addresses the temporal dimension: as the agent gets better across episodes, it needs less global exploration
- **Adaptive** addresses the spatial dimension: as a game unfolds and more is known, local exploration should drop

Together they achieve the best of both:

| Strategy | Win Rate | Std Dev |
|----------|----------|---------|
| Fixed | 56.33% | ±6.96% |
| Decay | 66.47% | ±0.80% |
| Adaptive | 67.27% | ±0.17% |
| **Decay + Adaptive** | **69.43%** | **±0.41%** |

**Best win rate AND second-lowest variance.**

---

## Slide 13 — Feature Upgrade: 7-dim → 10-dim

Alongside the ε improvements, we added three **logical inference features** using single-pass constraint propagation:

| Feature | Logic |
|---------|-------|
| **f8 — Inferred Safe** | =1 if a neighboring number cell has all its mines already flagged (this cell is provably safe) |
| **f9 — Inferred Mine** | =1 if a neighboring number cell's remaining hidden count equals its number (this cell is provably a mine) |
| **f10 — Continuous Danger** | Maximum residual mine probability across all neighboring constraints |

**Why this matters:**
- f8 and f9 give the agent **deterministic knowledge** — no learning needed, just logic
- f10 provides a soft danger score where f9 is ambiguous
- All computed in O(board_size) with per-state caching — zero training overhead

The 10-dim vector gives the agent an explicit reasoning layer on top of statistical learning.

---

## Slide 14 — Comprehensive Results

### Algorithm Comparison (Decay ε, 20,000 episodes × 3 seeds)

| Algorithm | Win Rate | Cond. Win Rate | Std Dev | Episode Time |
|-----------|----------|----------------|---------|-------------|
| **Q-Learning** | **69.50%** | 79.46% | ±0.28% | 10.1 ms |
| SARSA(λ) | 66.47% | 75.51% | ±0.80% | 11.9 ms |
| Monte Carlo | 55.60% | 63.80% | ±7.20% | 8.3 ms |
| Random Baseline | 0.00% | 0.00% | — | 0.05 ms |

**Key finding:** Q-Learning slightly outperforms SARSA(λ) (off-policy may suit this task better); Monte Carlo is unstable — high variance makes it unreliable.

### λ Ablation (SARSA, Decay+Adaptive ε, 3 seeds)

| λ | Win Rate | Cond. Win Rate |
|----|----------|----------------|
| 0.0 (one-step TD) | 69.50% ± 0.29% | 79.37% |
| 0.4 | 68.93% ± 0.66% | 79.08% |
| **0.8 (default)** | **69.43% ± 0.41%** | **79.48%** |
| 1.0 (near MC) | 67.83% ± 1.45% | 77.39% |

λ has limited effect in this domain — but λ=1.0 clearly hurts stability.

### Difficulty Scaling

| Board | Win Rate | Cond. Win Rate |
|-------|----------|----------------|
| 4×4 / 2 mines | 76.00% ± 2.89% | 86.97% |
| 6×6 / 5 mines | 62.47% ± 0.93% | 72.45% |
| 9×9 / 10 mines | 69.43% ± 0.41% | 79.48% |

First-click death (~12–14%) is consistent across all difficulties — purely a function of mine density.

---

## Slide 15 — Conclusion & Takeaways

**What we built:**
- A complete RL training pipeline: environment → features → algorithms → evaluation
- CLI + GUI interfaces for both human play and agent replay
- Five ablation experiments with reproducible results across 3 seeds

**What we learned:**

1. **Algorithm choice matters — but not as much as ε strategy.**
   The gap between Fixed and Decay+Adaptive (13 pp) dwarfs the gap between algorithms (~3 pp).

2. **Adaptive exploration beats global scheduling.**
   Responding to game state (what's revealed, what's deduced) is fundamentally more informative than episode number alone.

3. **Logic and learning are complementary.**
   Inference features (f8–f10) give the agent hard knowledge that statistical learning can only approximate.

4. **~12% of games are irreducible noise.**
   First-click death is not a model failure — it's a fundamental property of the game. Separating luck from skill (conditional win rate) is essential for honest evaluation.

**Best configuration:** SARSA(λ), λ=0.8, Decay+Adaptive ε, 10-dim features
→ **69.43% overall win rate / 79.48% conditional win rate** on 9×9 classic Minesweeper

---

*Thank you for listening. Questions?*
