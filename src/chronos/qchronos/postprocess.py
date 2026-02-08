from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def make_monotone_increasing(values: np.ndarray) -> np.ndarray:
    """
    把分位数序列修正为单调非降（便于概率插值）。

    背景：
    - 理论上分位数应该随 q 单调递增
    - 但实际模型输出可能因噪声出现轻微“穿插”
    - 若直接插值，可能得到负概率/不稳定结果

    做法：
    - 使用 cumulative max 强制单调（保守修正）
    """
    if values.ndim != 1:
        raise ValueError("values 必须为 1D 数组")
    return np.maximum.accumulate(values)


def _as_float_array(x: Sequence[float]) -> np.ndarray:
    return np.asarray([float(v) for v in x], dtype=float)


def event_prob_from_quantiles(
    *,
    quantile_levels: Sequence[float],
    quantile_values: Sequence[float],
    threshold: float,
    side: str,
    ensure_monotone: bool = True,
) -> float:
    """
    用“分位数网格”近似阈值事件概率。

    输入：
    - quantile_levels：分位点列表（例如 0.01..0.99）
    - quantile_values：对应分位点预测值（例如预测的 r_{t,N} 分位数）
    - threshold：阈值（和 quantile_values 同单位；例如 log-return 阈值 u=log(1+a)）
    - side：
      - "up"：计算 P(X >= threshold)
      - "down"：计算 P(X <= threshold)

    核心直觉（便于人工验证）：
    - Q_q 是一个数，使得“有 q 的概率落在它以下”
    - 若阈值落在 Q_{q*} 附近，说明大约 q* 的概率在阈值以下
      => 阈值以上概率约为 1 - q*

    注意：
    - quantile_levels 必须与 quantile_values 一一对应
    - quantile_levels 建议密集，否则概率会非常粗糙
    """
    levels = _as_float_array(quantile_levels)
    values = _as_float_array(quantile_values)
    if levels.shape != values.shape:
        raise ValueError("quantile_levels 与 quantile_values 长度不一致")

    # 排序：允许调用方传入无序 levels
    order = np.argsort(levels)
    levels = levels[order]
    values = values[order]

    if ensure_monotone:
        values = make_monotone_increasing(values)

    # 处理阈值落在网格之外的情况：直接截断到 0 或 1（clip）
    if threshold <= values[0]:
        # 阈值 <= 最小分位数：
        # - 上涨事件：几乎必然 >= threshold => 概率 ~ 1
        # - 下跌事件：几乎必然 <= threshold => 概率 ~ levels[0]（近似接近 0）
        q_star = float(levels[0])
    elif threshold >= values[-1]:
        q_star = float(levels[-1])
    else:
        # 插值：在 values 上找到 threshold 对应的分位点 q*
        q_star = float(np.interp(threshold, values, levels))

    if side == "up":
        return float(np.clip(1.0 - q_star, 0.0, 1.0))
    if side == "down":
        return float(np.clip(q_star, 0.0, 1.0))
    raise ValueError(f"side 只能是 'up' 或 'down'，得到: {side}")


@dataclass(frozen=True)
class ScoreSpec:
    """
    score 计算用到的一组字段名约定。

    说明：
    - 工程里常会有不同命名风格（q10_N / q0.1 / etc.）
    - 这里集中管理，避免散落到各脚本里导致难以人工核对
    """

    q10_col: str
    q50_col: str
    q90_col: str


def derive_interval_and_scores(
    df: pd.DataFrame,
    *,
    q10_col: str,
    q50_col: str,
    q90_col: str,
    p_up_col: str | None = None,
    p_down_col: str | None = None,
    eps: float = 1e-12,
    out_prefix: str = "",
) -> pd.DataFrame:
    """
    从区间分位数（q10/q50/q90）派生“区间宽度 + score1/score2/score3”。

    直觉解释（人工验证用）：
    - score1：只看“预期收益”（q50）
    - score2：收益/不确定性（同样收益，区间更窄=>更确定）
    - score3：事件概率差（若提供 p_up/p_down）
    """
    for col in (q10_col, q50_col, q90_col):
        if col not in df.columns:
            raise KeyError(f"缺少列: {col}")

    out = df.copy()
    width = (out[q90_col] - out[q10_col]).astype(float)
    out[f"{out_prefix}interval_width"] = width

    # score1：中位数收益（直接做横截面排序）
    out[f"{out_prefix}score1_med"] = out[q50_col].astype(float)

    # score2：不确定性折扣
    out[f"{out_prefix}score2_med_over_width"] = out[q50_col].astype(float) / (width + eps)

    # score3：事件概率差（若传入）
    if p_up_col is not None and p_down_col is not None:
        if p_up_col not in out.columns:
            raise KeyError(f"缺少列: {p_up_col}")
        if p_down_col not in out.columns:
            raise KeyError(f"缺少列: {p_down_col}")
        out[f"{out_prefix}score3_pdiff"] = out[p_up_col].astype(float) - out[p_down_col].astype(float)

    return out


def log_return_threshold_from_simple_return(a: float) -> float:
    """
    把简单收益阈值 a（例如 0.05）转成 log-return 阈值 u=log(1+a)。

    注意：
    - 下跌阈值通常用 d=log(1-a)，要求 a<1
    """
    a = float(a)
    return float(math.log(1.0 + a))


def log_return_down_threshold_from_simple_return(a: float) -> float:
    a = float(a)
    if a >= 1.0:
        raise ValueError("下跌阈值要求 a<1，否则 log(1-a) 无意义")
    return float(math.log(1.0 - a))

