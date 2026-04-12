"""
algorithm/test_agent.py

煙霧測試：確認算法模組可以和環境正常對接並訓練。
執行方式：python3 algorithm/test_agent.py
"""

import sys
import random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment import MinesweeperEnv
from algorithm.agent import SARSALambdaAgent
from algorithm.features import extract_features, FEATURE_DIM

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    msg = f"  {tag}  {name}"
    if not cond and detail:
        msg += f"\n         {detail}"
    print(msg)
    results.append(cond)


# ── 環境 ──────────────────────────────────────────────────
env = MinesweeperEnv(grid_size=(9, 9), num_mines=10)


# ======================================================
# 1. 特徵萃取
# ======================================================
print("\n--- 1. 特徵萃取 ---")

state = env.reset()
valid = env.get_valid_actions()
phi = extract_features(state, valid[0])

check("特徵向量長度等於 FEATURE_DIM", len(phi) == FEATURE_DIM,
      f"實際長度={len(phi)}")
check("特徵向量型別是 np.ndarray", isinstance(phi, np.ndarray))
check("特徵值全部有限（無 NaN / Inf）", np.all(np.isfinite(phi)),
      f"phi={phi}")
check("特徵值範圍合理（-0.1 ~ 1.1）",
      np.all(phi >= -0.1) and np.all(phi <= 1.1),
      f"phi={phi}")


# ======================================================
# 2. Agent 初始化
# ======================================================
print("\n--- 2. Agent 初始化 ---")

agent = SARSALambdaAgent()

check("w 長度等於 FEATURE_DIM", len(agent.w) == FEATURE_DIM)
check("e 長度等於 FEATURE_DIM", len(agent.e) == FEATURE_DIM)
check("初始 w 全為 0", np.all(agent.w == 0))
check("初始 e 全為 0", np.all(agent.e == 0))


# ======================================================
# 3. select_action
# ======================================================
print("\n--- 3. select_action ---")

state = env.reset()
valid = env.get_valid_actions()
action = agent.select_action(state, valid)

check("select_action 回傳 tuple", isinstance(action, tuple))
check("回傳動作在合法清單內", action in valid,
      f"action={action}")
check("回傳動作是二維座標", len(action) == 2)

# ε=1.0 時應該純隨機
agent.epsilon = 1.0
actions_seen = set(agent.select_action(state, valid) for _ in range(30))
check("ε=1.0 時有隨機性（30 次至少出現 2 種動作）",
      len(actions_seen) > 1)

# ε=0.0 時應該純貪婪（同樣 state 每次一樣）
agent.epsilon = 0.0
greedy_actions = [agent.select_action(state, valid) for _ in range(5)]
check("ε=0.0 時每次選同一個動作", len(set(greedy_actions)) == 1)

agent.epsilon = 0.3  # 還原


# ======================================================
# 4. update（不崩潰、w 有更新）
# ======================================================
print("\n--- 4. update ---")

state = env.reset()
valid = env.get_valid_actions()
action = agent.select_action(state, valid)
w_before = agent.w.copy()

next_state, reward, done, info = env.step(action)

if done:
    agent.update(state, action, reward, None, None, done=True)
    check("done=True 時 update 不崩潰", True)
else:
    next_valid = env.get_valid_actions()
    next_action = agent.select_action(next_state, next_valid)
    agent.update(state, action, reward, next_state, next_action, done=False)
    check("done=False 時 update 不崩潰", True)

check("update 後 w 有改變", not np.all(agent.w == w_before))
check("update 後 w 全部有限", np.all(np.isfinite(agent.w)))


# ======================================================
# 5. reset_trace
# ======================================================
print("\n--- 5. reset_trace ---")

agent.e = np.ones(FEATURE_DIM)
w_before = agent.w.copy()
agent.reset_trace()

check("reset_trace 後 e 全為 0", np.all(agent.e == 0))
check("reset_trace 後 w 不變", np.all(agent.w == w_before))


# ======================================================
# 6. 完整一局（不崩潰）
# ======================================================
print("\n--- 6. 完整一局 ---")

agent = SARSALambdaAgent()
crashed = False
try:
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
except Exception as e:
    crashed = True
    print(f"         崩潰：{e}")

check("完整一局不崩潰", not crashed)


# ======================================================
# 7. 短訓練 100 局（w 有學到東西）
# ======================================================
print("\n--- 7. 短訓練 100 局 ---")

agent = SARSALambdaAgent()
w_before = agent.w.copy()
crashed = False

try:
    for ep in range(100):
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

        agent.epsilon = max(0.05, agent.epsilon * 0.995)

except Exception as e:
    crashed = True
    print(f"         崩潰：{e}")

check("100 局訓練不崩潰", not crashed)
check("訓練後 w 有改變（agent 有在學習）",
      not np.all(agent.w == w_before))
check("訓練後 w 全部有限（無爆炸）",
      np.all(np.isfinite(agent.w)))


# ======================================================
# 彙總
# ======================================================
total = len(results)
passed = sum(results)
print(f"\n{'='*40}")
print(f"結果：{passed}/{total} 通過")
if passed == total:
    print("全部通過，算法模組可以正常運作。")
else:
    print(f"有 {total - passed} 項失敗，請檢查上方 [FAIL] 條目。")
print("=" * 40)

sys.exit(0 if passed == total else 1)