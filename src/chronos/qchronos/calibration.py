from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


def conformal_nonconformity(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Conformal 区间校准用的 nonconformity 分数：

    a_t = max(lower - y, y - upper, 0)

    直觉：
    - 若真实值落在区间内 => a_t = 0
    - 若真实值落在区间外 => a_t = 距离最近边界的“跑出幅度”
    """
    y = np.asarray(y_true, dtype=float)
    l = np.asarray(lower, dtype=float)
    u = np.asarray(upper, dtype=float)
    return np.maximum.reduce([l - y, y - u, np.zeros_like(y)])


def rolling_conformal_q(
    scores: np.ndarray,
    *,
    alpha: float,
    window: int | None = None,
    min_count: int = 30,
) -> np.ndarray:
    """
    计算滚动的 conformal 扩张量 q_a（每个时间点一个）。

    输入：
    - scores：按时间顺序的 nonconformity 分数（来自 calib 段）
    - alpha：风险水平；例如目标覆盖 80% => alpha=0.2
    - window：滚动窗口长度；None 表示 expanding（从起点累计）

    输出：
    - q：与 scores 等长；前期样本不足时为 NaN
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha 必须在 (0,1) 内")
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    q = np.full(n, np.nan, dtype=float)

    for t in range(n):
        start = 0 if window is None else max(0, t - window + 1)
        window_scores = scores[start : t + 1]
        window_scores = window_scores[np.isfinite(window_scores)]
        if window_scores.size < min_count:
            continue
        q[t] = float(np.quantile(window_scores, 1.0 - alpha))
    return q


def apply_conformal_expansion(lower_raw: np.ndarray, upper_raw: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    应用 conformal 扩张：
    - lower_cal = lower_raw - q
    - upper_cal = upper_raw + q
    """
    l = np.asarray(lower_raw, dtype=float)
    u = np.asarray(upper_raw, dtype=float)
    qv = np.asarray(q, dtype=float)
    return l - qv, u + qv


@dataclass
class ProbabilityCalibrator:
    """
    概率校准器：把 raw 概率映射到更“可靠”的 cal 概率。

    支持两类常见方法：
    - isotonic：单调非参数（更灵活）
    - platt：sigmoid（逻辑回归；更平滑、参数更少）

    注意：
    - 这里只实现最小可用封装，便于人工验证
    - 校准必须严格在 OOS 的 calib 段拟合，再在 test 段评估
    """

    method: Literal["isotonic", "platt"] = "isotonic"

    # 训练后对象（延迟导入 sklearn，避免硬依赖）
    _model: object | None = None

    def fit(self, p_raw: np.ndarray, y_event: np.ndarray) -> "ProbabilityCalibrator":
        p = np.asarray(p_raw, dtype=float)
        y = np.asarray(y_event, dtype=int)
        if p.shape[0] != y.shape[0]:
            raise ValueError("p_raw 与 y_event 长度不一致")

        # clip：避免 0/1 导致 logit 溢出或数值不稳定
        p = np.clip(p, 1e-6, 1.0 - 1e-6)

        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(p, y)
            self._model = model
            return self

        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression

            # Platt scaling：用 logits 或直接用 p 也行，这里用 logit(p) 更贴近经典做法
            x = np.log(p / (1.0 - p)).reshape(-1, 1)
            model = LogisticRegression(solver="lbfgs")
            model.fit(x, y)
            self._model = model
            return self

        raise ValueError(f"未知 method: {self.method}")

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("ProbabilityCalibrator 尚未 fit")
        p = np.asarray(p_raw, dtype=float)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)

        if self.method == "isotonic":
            # IsotonicRegression.predict
            return np.asarray(self._model.predict(p), dtype=float)  # type: ignore[no-any-return]

        if self.method == "platt":
            x = np.log(p / (1.0 - p)).reshape(-1, 1)
            proba = self._model.predict_proba(x)  # type: ignore[union-attr]
            return np.asarray(proba[:, 1], dtype=float)

        raise ValueError(f"未知 method: {self.method}")


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """
    Brier score：概率预测的均方误差。
    - p：预测概率
    - y：事件真值（0/1）
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def reliability_table(p: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> pd.DataFrame:
    """
    构造 reliability diagram 的“数表版”（便于人工验证/打印）。

    输出列：
    - bin_left / bin_right：概率分箱区间
    - n：该箱样本量
    - p_mean：预测概率均值
    - y_freq：真实发生频率
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    p = np.clip(p, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(p, bins, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            rows.append(
                dict(
                    bin_left=float(bins[b]),
                    bin_right=float(bins[b + 1]),
                    n=0,
                    p_mean=np.nan,
                    y_freq=np.nan,
                )
            )
            continue
        rows.append(
            dict(
                bin_left=float(bins[b]),
                bin_right=float(bins[b + 1]),
                n=int(mask.sum()),
                p_mean=float(np.mean(p[mask])),
                y_freq=float(np.mean(y[mask])),
            )
        )

    return pd.DataFrame(rows)

