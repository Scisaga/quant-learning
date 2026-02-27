from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StaticGateConfig:
    """
    静态门控（结构约束）配置。

    说明：
    - Static gate 的目的不是“提高收益”，而是先把不可交易/数据不可信的样本剔除掉
    - 这一步越清晰，后面的 IC/回测越不容易被脏样本误导
    """

    min_turnover_20: float | None = None  # 近 20 日成交额均值下限（若提供）
    min_data_coverage: float = 0.95  # context 窗口有效数据比例（0~1）


def compute_static_pass(
    df: pd.DataFrame,
    *,
    turnover_20_col: str | None = None,
    data_coverage_col: str | None = None,
    tradeable_col: str | None = None,
    cfg: StaticGateConfig | None = None,
    out_col: str = "static_pass",
) -> pd.DataFrame:
    """
    计算静态门控 `static_pass`。

    约定：
    - df 是“逐日逐股”的表（索引可为 MultiIndex(datetime, instrument) 或普通 index）
    - 你可以按你的数据情况选择传入哪些列；未提供的条件不会生效

    输出：
    - 新增布尔列 out_col
    """
    if cfg is None:
        cfg = StaticGateConfig()

    out = df.copy()
    static_ok = pd.Series(True, index=out.index)

    if tradeable_col is not None and tradeable_col in out.columns:
        static_ok &= out[tradeable_col].astype(bool)

    if data_coverage_col is not None and data_coverage_col in out.columns:
        static_ok &= out[data_coverage_col].astype(float) >= float(cfg.min_data_coverage)

    if cfg.min_turnover_20 is not None and turnover_20_col is not None and turnover_20_col in out.columns:
        static_ok &= out[turnover_20_col].astype(float) >= float(cfg.min_turnover_20)

    out[out_col] = static_ok.astype(bool)
    return out


@dataclass(frozen=True)
class DynamicGateConfig:
    """
    动态门控（信号触发）配置。

    字段直觉：
    - p0：上涨阈值概率要足够大（胜率）
    - w0：区间宽度不能太大（不确定性约束）
    - m0：中位数收益要为正/足够大（收益门槛）
    - q0：下跌阈值概率过高则 risk-off（下行风险）
    - s0：左尾（q10）太差则 risk-off（尾部风险）
    """

    p0: float = 0.6
    w0: float = 0.20
    m0: float = 0.0
    q0: float = 0.6
    s0: float = 0.05


def compute_dynamic_signals(
    df: pd.DataFrame,
    *,
    p_up_col: str,
    p_down_col: str,
    q10_col: str,
    q50_col: str,
    width_col: str,
    static_pass_col: str = "static_pass",
    cfg: DynamicGateConfig | None = None,
    out_long_col: str = "signal_long",
    out_risk_off_col: str = "signal_risk_off",
) -> pd.DataFrame:
    """
    计算动态信号：`signal_long` 与 `signal_risk_off`。

    注意：
    - 这里不直接给“仓位大小”，只给触发信号，便于你在回测里再做仓位管理
    - 如果你做多空，可以在此基础上扩展 `signal_short`
    """
    if cfg is None:
        cfg = DynamicGateConfig()

    missing = [c for c in (p_up_col, p_down_col, q10_col, q50_col, width_col) if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要列: {missing}")

    out = df.copy()

    static_pass = out[static_pass_col].astype(bool) if static_pass_col in out.columns else True

    p_up = out[p_up_col].astype(float)
    p_down = out[p_down_col].astype(float)
    q10 = out[q10_col].astype(float)
    q50 = out[q50_col].astype(float)
    width = out[width_col].astype(float)

    # 多头触发：胜率高 + 不确定性小 + 预期收益为正
    long_signal = (p_up >= cfg.p0) & (width <= cfg.w0) & (q50 >= cfg.m0)

    # 风控触发：下行概率高 或 左尾很差
    risk_off = (p_down >= cfg.q0) | (q10 <= -cfg.s0)

    # 静态门控：若不通过，则所有动态信号都应为 False
    long_signal = long_signal & static_pass
    risk_off = risk_off & static_pass

    out[out_long_col] = long_signal.astype(bool)
    out[out_risk_off_col] = risk_off.astype(bool)
    return out

