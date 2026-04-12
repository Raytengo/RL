import numpy as np

HIDDEN = -1
FEATURE_DIM = 7  # 這個數字如果改了，agent.py 會自動跟著變


def extract_features(state: dict, action: tuple) -> np.ndarray:
    grid = state["map"]
    rows, cols = state["grid_size"]
    r, c = action

    # ── 收集這格周圍 8 個鄰居的值 ──
    neighbor_vals = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor_vals.append(grid[nr][nc])

    # 分類鄰居
    hidden_neighbors = [v for v in neighbor_vals if v == HIDDEN]
    number_neighbors = [v for v in neighbor_vals if 0 <= v <= 8]

    # ── 計算七個特徵 ──

    # f1: 周圍未翻開格數（資訊量的反指標）
    f1 = len(hidden_neighbors) / 8.0

    # f2: 周圍數字格總和（絕對壓力）
    f2 = sum(number_neighbors) / 64.0  # 最大值是 8 格 × 數字 8 = 64

    # f3: 危險度估計（核心特徵）
    # 數字總和 ÷ 未翻開格數，除以 8 做正規化
    denom = max(len(hidden_neighbors), 1)  # 避免除以零
    f3 = (sum(number_neighbors) / denom) / 8.0

    # f4: 完全孤立（周圍沒有任何數字提示）
    f4 = 1.0 if len(number_neighbors) == 0 else 0.0

    # f5: 全局未翻開格佔比（遊戲進度）
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

    return np.array([f1, f2, f3, f4, f5, f6, f7], dtype=np.float64)