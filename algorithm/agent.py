import random
import numpy as np
from .features import extract_features, FEATURE_DIM


class SARSALambdaAgent:
    def __init__(
        self,
        alpha: float = 0.01,   # 學習率
        gamma: float = 0.99,   # 折扣因子
        lam: float = 0.8,      # λ，trace 衰減速度
        epsilon: float = 0.3,  # ε-greedy 初始探索率
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon

        self.w = np.zeros(FEATURE_DIM)  # 學習參數，跨局保留
        self.e = np.zeros(FEATURE_DIM)  # Eligibility trace，每局清零

    # ── 對外三個方法，訓練迴圈只需要這三個 ──────────────

    def select_action(self, state: dict, valid_actions: list) -> tuple:
        """ε-greedy 選動作，action masking 自動完成"""
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        return max(valid_actions, key=lambda a: self._Q(state, a))

    def update(self, s, a, r, s_next, a_next, done: bool):
        """SARSA(λ) 單步更新，done=True 時 s_next 和 a_next 傳 None"""
        phi = extract_features(s, a)

        # 步驟一：算誤差 δ
        if done:
            delta = r - self.w @ phi
        else:
            delta = r + self.gamma * self._Q(s_next, a_next) - self.w @ phi

        # 步驟二：更新 trace
        self.e = self.gamma * self.lam * self.e + phi

        # 步驟三：更新權重（唯一真正學習的地方）
        self.w += self.alpha * delta * self.e

    def reset_trace(self):
        """每局開始前呼叫，清除 trace，w 不動"""
        self.e = np.zeros(FEATURE_DIM)

    # ── 內部方法 ─────────────────────────────────────────

    def _Q(self, state: dict, action: tuple) -> float:
        return float(self.w @ extract_features(state, action))