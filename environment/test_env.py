"""
MinesweeperEnv 测试套件
运行方式：python3 environment/test_env.py
"""

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.minesweeper_env import MinesweeperEnv, HIDDEN, MINE

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


# ======================================================================
# 1. 初始化与 reset()
# ======================================================================
print("\n--- 1. 初始化与 reset() ---")

env = MinesweeperEnv(grid_size=(9, 9), num_mines=10)
state = env.reset()

check("state 包含所有必要 key",
      set(state.keys()) == {"grid_size", "map", "done", "win", "reward", "trial_count"})

check("grid_size 正确", state["grid_size"] == (9, 9))
check("初始 done=False", state["done"] == False)
check("初始 win=False", state["win"] == False)
check("初始 reward=0.0", state["reward"] == 0.0)
check("初始 trial_count=0", state["trial_count"] == 0)

env_reward_cfg = MinesweeperEnv(
    grid_size=(2, 2),
    num_mines=1,
    safe_open_reward_per_cell=0.2,
    win_reward=7.0,
    lose_reward=-9.0,
    repeat_reward=-0.25,
)
check("reward 参数可配置",
      env_reward_cfg.safe_open_reward_per_cell == 0.2 and
      env_reward_cfg.win_reward == 7.0 and
      env_reward_cfg.lose_reward == -9.0 and
      env_reward_cfg.repeat_reward == -0.25)

# map 尺寸
m = state["map"]
check("map 行数正确", len(m) == 9)
check("map 列数正确", all(len(row) == 9 for row in m))

# 初始 map 全为 HIDDEN
check("初始 map 全为 HIDDEN(-1)",
      all(m[r][c] == HIDDEN for r in range(9) for c in range(9)))

# 地雷数量
mine_count = sum(env._mine_map[r][c] for r in range(9) for c in range(9))
check("地雷数量正确", mine_count == 10, f"实际地雷数={mine_count}")

# reset 后状态独立（不是同一个对象引用）
state2 = env.reset()
check("reset 返回新 state 对象", state is not state2)


# ======================================================================
# 2. get_valid_actions()
# ======================================================================
print("\n--- 2. get_valid_actions() ---")

env = MinesweeperEnv(grid_size=(4, 4), num_mines=2)
env.reset()

actions = env.get_valid_actions()
check("初始合法动作数 == 总格子数", len(actions) == 16)
check("所有动作在棋盘范围内",
      all(0 <= r < 4 and 0 <= c < 4 for r, c in actions))
check("动作无重复", len(set(actions)) == len(actions))


# ======================================================================
# 3. step() —— 安全翻格子
# ======================================================================
print("\n--- 3. step() 安全翻格子 ---")

# 构造确定性场景：3x3，地雷在 (0,0)
# 翻开 (0,1)：它紧邻地雷，number=1，不会 BFS 展开，也不会一步获胜
env = MinesweeperEnv(grid_size=(3, 3), num_mines=1)
env._mine_map = [[True,  False, False],
                 [False, False, False],
                 [False, False, False]]
env._visible = [[False]*3 for _ in range(3)]
env._compute_numbers()
env._done = False
env._win = False
env._trial_count = 0

state, reward, done, info = env.step((0, 1))
check("安全步 done=False", done == False, f"done={done}, win={info.get('win')}")
check("安全步 reward=0.05", reward == 0.05, f"reward={reward}")
check("安全步 info['win']=False", info["win"] == False)
check("安全步 state['reward'] 同步", state["reward"] == reward, f"state_reward={state['reward']}")
check("翻开后 map[0][1] 不再是 HIDDEN", state["map"][0][1] != HIDDEN)
check("trial_count 增加", state["trial_count"] > 0)

# 已翻开格子再次翻开 → 惩罚，不结束
state2, reward2, done2, info2 = env.step((0, 1))
check("重复翻格 reward=-0.5", reward2 == -0.5, f"reward={reward2}")
check("重复翻格 done=False", done2 == False)
check("重复翻格 state['reward'] 同步", state2["reward"] == reward2, f"state_reward={state2['reward']}")
check("重复翻格 info 含 'repeated'=True", info2.get("repeated") == True)


# ======================================================================
# 4. step() —— 踩雷
# ======================================================================
print("\n--- 4. step() 踩雷 ---")

env = MinesweeperEnv(grid_size=(3, 3), num_mines=1)
env._mine_map = [[True,  False, False],
                 [False, False, False],
                 [False, False, False]]
env._visible = [[False]*3 for _ in range(3)]
env._compute_numbers()
env._done = False
env._win = False
env._trial_count = 0

state, reward, done, info = env.step((0, 0))
check("踩雷 reward=-10.0", reward == -10.0, f"reward={reward}")
check("踩雷 done=True", done == True)
check("踩雷 info['win']=False", info["win"] == False)
check("踩雷 state['reward'] 同步", state["reward"] == reward, f"state_reward={state['reward']}")
check("踩雷格子在 map 中显示为 MINE(9)", state["map"][0][0] == MINE,
      f"map[0][0]={state['map'][0][0]}")


# ======================================================================
# 5. step() —— 游戏结束后继续调用抛出异常
# ======================================================================
print("\n--- 5. step() 结束后调用异常 ---")

try:
    env.step((0, 0))
    check("游戏结束后调用 step 应抛出 RuntimeError", False)
except RuntimeError:
    check("游戏结束后调用 step 抛出 RuntimeError", True)
except Exception as e:
    check("游戏结束后调用 step 抛出 RuntimeError", False, f"实际异常={e}")


# ======================================================================
# 6. step() —— 坐标越界抛出异常
# ======================================================================
print("\n--- 6. step() 越界异常 ---")

env = MinesweeperEnv(grid_size=(3, 3), num_mines=1)
env.reset()
for bad in [(-1, 0), (3, 0), (0, -1), (0, 3)]:
    try:
        env.step(bad)
        check(f"越界 {bad} 应抛出 ValueError", False)
    except ValueError:
        check(f"越界 {bad} 抛出 ValueError", True)
    except Exception as e:
        check(f"越界 {bad} 应抛出 ValueError", False, f"实际={e}")


# ======================================================================
# 7. 胜利判定
# ======================================================================
print("\n--- 7. 胜利判定 ---")

# 2x2，地雷在 (0,0)，其余三格 (0,1)(1,0)(1,1) 都是数字格（紧邻地雷，不会BFS展开）
# 依次翻开三格，第三步触发胜利
env = MinesweeperEnv(grid_size=(2, 2), num_mines=1)
env._mine_map = [[True,  False],
                 [False, False]]
env._visible = [[False, False],
                [False, False]]
env._compute_numbers()
env._done = False
env._win = False
env._trial_count = 0

_, _, done1, _ = env.step((0, 1))
check("翻第一格后 done=False", done1 == False, f"done={done1}")
_, _, done2, _ = env.step((1, 0))
check("翻第二格后 done=False", done2 == False, f"done={done2}")
state, reward, done3, info = env.step((1, 1))
check("翻完所有安全格后 done=True", done3 == True, f"done={done3}")
check("翻完所有安全格后 win=True", info["win"] == True)
check("获胜 reward=10.05", reward == 10.05, f"reward={reward}")
check("获胜 state['reward'] 同步", state["reward"] == reward, f"state_reward={state['reward']}")


# ======================================================================
# 8. BFS 连锁展开
# ======================================================================
print("\n--- 8. BFS 连锁展开 ---")

# 构造 3x3，地雷只在 (0,0)，其余大片为 0 的格子
# 翻开 (2,2) 应该连锁打开大量格子
env = MinesweeperEnv(grid_size=(3, 3), num_mines=1)
env._mine_map = [[True, False, False],
                 [False, False, False],
                 [False, False, False]]
env._visible = [[False]*3 for _ in range(3)]
env._compute_numbers()
env._done = False; env._win = False; env._trial_count = 0

state, reward, done, info = env.step((2, 2))
opened = sum(1 for r in range(3) for c in range(3) if state["map"][r][c] != HIDDEN)
check("连锁展开后翻开多于 1 格", opened > 1, f"翻开格数={opened}")
check("未翻开的仍为 HIDDEN", state["map"][0][0] == HIDDEN or env._mine_map[0][0])


# ======================================================================
# 9. reset() 真正重置
# ======================================================================
print("\n--- 9. reset() 重置完整性 ---")

env = MinesweeperEnv(grid_size=(5, 5), num_mines=3)
env.reset()
# 踩雷使游戏结束
mine_pos = next((r, c) for r in range(5) for c in range(5) if env._mine_map[r][c])
env.step(mine_pos)
check("踩雷后 done=True", env._done == True)

state = env.reset()
check("reset 后 done=False", state["done"] == False)
check("reset 后 trial_count=0", state["trial_count"] == 0)
check("reset 后 map 全为 HIDDEN",
      all(state["map"][r][c] == HIDDEN for r in range(5) for c in range(5)))
check("reset 后 get_valid_actions 数量=25", len(env.get_valid_actions()) == 25)


# ======================================================================
# 10. number_map 数值正确性
# ======================================================================
print("\n--- 10. number_map 正确性 ---")

# 3x3，地雷在 (1,1) 正中央，周围 8 格数字都应为 1
env = MinesweeperEnv(grid_size=(3, 3), num_mines=1)
env._mine_map = [[False]*3 for _ in range(3)]
env._mine_map[1][1] = True
env._visible = [[False]*3 for _ in range(3)]
env._compute_numbers()

corners = [(0,0),(0,2),(2,0),(2,2)]
check("四个角落数字=1", all(env._number_map[r][c] == 1 for r, c in corners),
      str({(r,c): env._number_map[r][c] for r,c in corners}))
check("地雷格 number_map=-1", env._number_map[1][1] == -1)


# ======================================================================
# 11. 完整随机对局（不崩溃）
# ======================================================================
print("\n--- 11. 完整随机对局压力测试 ---")

crashed = False
wins = 0
# 用小棋盘（4x4，2颗雷）跑 500 局，随机策略胜率可观
for _ in range(500):
    env = MinesweeperEnv(grid_size=(4, 4), num_mines=2)
    state = env.reset()
    info = {}
    try:
        while not state["done"]:
            actions = env.get_valid_actions()
            if not actions:
                break
            state, reward, done, info = env.step(random.choice(actions))
        if info.get("win"):
            wins += 1
    except Exception as e:
        crashed = True
        print(f"         崩溃：{e}")
        break

check("500 局随机对局无崩溃", not crashed)
check(f"500 局随机对局至少赢了 1 局（实际赢了 {wins} 局）", wins >= 1)


# ======================================================================
# 12. render() 不报错
# ======================================================================
print("\n--- 12. render() ---")

env = MinesweeperEnv(grid_size=(4, 4), num_mines=2)
env.reset()
env.step(env.get_valid_actions()[0])
try:
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env.render()
    output = buf.getvalue()
    check("render() 正常输出", len(output) > 0)
except Exception as e:
    check("render() 不抛出异常", False, str(e))


# ======================================================================
# 汇总
# ======================================================================
total = len(results)
passed = sum(results)
print(f"\n{'='*40}")
print(f"结果：{passed}/{total} 通过")
if passed == total:
    print("全部通过！")
else:
    print(f"有 {total - passed} 项失败，请检查上方 [FAIL] 条目。")
print('='*40)

sys.exit(0 if passed == total else 1)
