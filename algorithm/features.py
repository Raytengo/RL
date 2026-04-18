import numpy as np

HIDDEN = -1
FEATURE_DIM = 10  # 這個數字如果改了，agent.py 會自動跟著變

# ── 推理輔助函數 ──────────────────────────────────────────

def _get_neighbors(r, c, rows, cols):
    """回傳 (r,c) 的所有合法鄰居座標"""
    result = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result.append((nr, nc))
    return result


def _infer_mines(grid, rows, cols):
    """
    單遍掃描：如果某個數字格的值 == 周圍未翻開格數，
    那周圍所有未翻開格都是地雷。
    """
    mine_set = set()
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if not (1 <= v <= 8):
                continue
            neighbors = _get_neighbors(r, c, rows, cols)
            hidden = [n for n in neighbors if grid[n[0]][n[1]] == HIDDEN]
            if len(hidden) == v:
                mine_set.update(hidden)
    return mine_set


def _infer_safe(grid, rows, cols, mine_set):
    """
    單遍掃描：如果某個數字格的值 == 周圍推理地雷數，
    那周圍剩餘的未翻開格都安全。
    """
    safe_set = set()
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if not (1 <= v <= 8):
                continue
            neighbors = _get_neighbors(r, c, rows, cols)
            hidden = [n for n in neighbors if grid[n[0]][n[1]] == HIDDEN]
            known_mines = [n for n in neighbors if n in mine_set]
            if len(known_mines) == v:
                for n in hidden:
                    if n not in mine_set:
                        safe_set.add(n)
    return safe_set


# ── 每個 state 只算一次推理結果（簡易 cache） ────────────

_cache_key = None
_cache_mine_set = None
_cache_safe_set = None


def _get_inference(state):
    """
    對同一個 state（以 map tuple 為 key）只跑一次推理，
    避免 select_action 對每個 valid action 重複掃描棋盤。
    """
    global _cache_key, _cache_mine_set, _cache_safe_set

    grid = state["map"]
    rows, cols = state["grid_size"]
    key = (rows, cols, tuple(grid[r][c] for r in range(rows) for c in range(cols)))

    if key != _cache_key:
        mine_set = _infer_mines(grid, rows, cols)
        safe_set = _infer_safe(grid, rows, cols, mine_set)
        _cache_key = key
        _cache_mine_set = mine_set
        _cache_safe_set = safe_set

    return _cache_mine_set, _cache_safe_set


# ── 主特徵函數 ───────────────────────────────────────────

def extract_features(state: dict, action: tuple) -> np.ndarray:
    grid = state["map"]
    rows, cols = state["grid_size"]
    r, c = action

    # ── 收集這格周圍 8 個鄰居的值 ──
    neighbors = _get_neighbors(r, c, rows, cols)
    neighbor_vals = [grid[nr][nc] for nr, nc in neighbors]

    hidden_neighbors = [v for v in neighbor_vals if v == HIDDEN]
    number_neighbors = [v for v in neighbor_vals if 1 <= v <= 8]
    # 注意：數字 0 的格子周圍無雷，對 f2/f3 貢獻為 0，不排除

    # ── 原有七個特徵 ──

    # f1: 周圍未翻開格比例
    f1 = len(hidden_neighbors) / 8.0

    # f2: 周圍數字格總和
    f2 = sum(number_neighbors) / 64.0

    # f3: 危險度估計（數字總和 / 未翻開格數 / 8）
    denom = max(len(hidden_neighbors), 1)
    f3 = (sum(number_neighbors) / denom) / 8.0

    # f4: 完全孤立（周圍沒有任何數字提示）
    all_number_neighbors = [v for v in neighbor_vals if 0 <= v <= 8]
    f4 = 1.0 if len(all_number_neighbors) == 0 else 0.0

    # f5: 全局未翻開格佔比
    total_cells = rows * cols
    hidden_count = sum(
        grid[rr][cc] == HIDDEN
        for rr in range(rows)
        for cc in range(cols)
    )
    f5 = hidden_count / total_cells

    # f6: 全局已知地雷佔比（踩了幾顆雷）
    mine_count = sum(
        grid[rr][cc] == 9
        for rr in range(rows)
        for cc in range(cols)
    )
    f6 = mine_count / total_cells

    # f7: Bias（固定為 1.0，讓線性模型有截距）
    f7 = 1.0

    # ── 三個新特徵（推理結果） ──

    mine_set, safe_set = _get_inference(state)

    # f8: 這格被推理確定為安全？
    f8 = 1.0 if (r, c) in safe_set else 0.0

    # f9: 這格被推理確定為地雷？
    f9 = 1.0 if (r, c) in mine_set else 0.0

    # f10: 連續危險度
    # 對每個鄰居數字格 n，算 (n.數字 - n.周圍推理地雷數) / n.周圍未翻開格數
    # 值越接近 1.0 表示這格越可能是雷，越接近 0 越安全
    danger_scores = []
    for nr, nc in neighbors:
        nv = grid[nr][nc]
        if not (1 <= nv <= 8):
            continue
        n_neighbors = _get_neighbors(nr, nc, rows, cols)
        n_hidden = [nn for nn in n_neighbors if grid[nn[0]][nn[1]] == HIDDEN]
        n_mines = [nn for nn in n_neighbors if nn in mine_set]
        remaining_mines = nv - len(n_mines)
        remaining_hidden = len(n_hidden) - len(n_mines)
        if remaining_hidden > 0:
            danger_scores.append(remaining_mines / remaining_hidden)
    f10 = max(danger_scores) if danger_scores else 0.5

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dtype=np.float64)
