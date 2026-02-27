from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


def build_calendar_known_future_features(
    timestamps: pd.DatetimeIndex,
    *,
    prefix: str = "cal_",
) -> pd.DataFrame:
    """
    构造 known-future（日历型）协变量。

    为什么它属于 known-future？
    - 交易日历/星期几/月末等信息，在今天就能“确定未来会是什么”，不涉及信息泄漏。

    输出：
    - index: timestamps
    - columns: 一组 0/1 或整数特征，列名带 prefix
    """
    ts = pd.DatetimeIndex(timestamps)
    # 说明：这里用 pandas 自带的日历属性，便于人工验证（可直接打印看是否符合直觉）
    df = pd.DataFrame(index=ts)
    df[f"{prefix}dow"] = ts.dayofweek.astype("int16")  # 0=周一 ... 4=周五
    df[f"{prefix}dom"] = ts.day.astype("int16")  # day of month
    df[f"{prefix}month"] = ts.month.astype("int16")
    df[f"{prefix}is_month_end"] = ts.is_month_end.astype("int8")
    df[f"{prefix}is_quarter_end"] = ts.is_quarter_end.astype("int8")
    df[f"{prefix}is_year_end"] = ts.is_year_end.astype("int8")
    return df


def _safe_log(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    # 金融价格可能出现 0（脏数据），这里做一个非常保守的保护
    return np.log(np.maximum(x.astype(float).to_numpy(), eps))


def build_past_only_features_from_close(
    close: pd.Series,
    *,
    windows: tuple[int, ...] = (5, 10, 20, 60),
    prefix: str = "feat_",
) -> pd.DataFrame:
    """
    从收盘价构造一组“past-only”数值特征（形态学的最小集合）。

    输入：
    - close：pd.Series，index 为 DatetimeIndex（或可转），值为价格

    输出：
    - DataFrame(index=close.index)，列带 prefix

    特征选择原则（便于人工验证）：
    - 只用历史滚动统计，不用任何未来信息
    - 先覆盖：趋势（动量）/ 波动 / 回撤（尾部） 这三类最关键维度
    """
    close = close.dropna()
    if close.empty:
        raise ValueError("close 为空，无法构造特征")

    close = close.astype(float)
    idx = pd.DatetimeIndex(close.index)

    # 价格/收益口径：这里统一用 log-return 更稳定
    logp = pd.Series(_safe_log(close), index=idx, name="logp")
    r1 = logp.diff(1).rename("logret_1")

    out = pd.DataFrame(index=idx)
    out[f"{prefix}logp"] = logp
    out[f"{prefix}logret_1"] = r1

    for w in windows:
        w = int(w)
        # 动量：过去 w 日累计 log-return（近似累计收益）
        out[f"{prefix}mom_{w}"] = logp.diff(w)

        # 波动：过去 w 日 log-return 标准差
        out[f"{prefix}vol_{w}"] = r1.rolling(w, min_periods=max(3, w // 3)).std()

        # 回撤：过去 w 日从窗口最高点回撤幅度（用价格空间更直觉）
        rolling_max = close.rolling(w, min_periods=max(3, w // 3)).max()
        out[f"{prefix}dd_{w}"] = 1.0 - close / rolling_max

    # 额外：布林带宽度（用 20 日为默认，便于人工理解）
    if 20 in windows:
        ma20 = close.rolling(20, min_periods=10).mean()
        sd20 = close.rolling(20, min_periods=10).std()
        out[f"{prefix}bb_width_20"] = (4.0 * sd20) / ma20  # (upper-lower)/ma, upper/lower=±2σ

    return out


@dataclass(frozen=True)
class RegimeConfig:
    """
    Regime 标签配置。

    默认做法：trend(上/震荡/下) × vol(高/低) => 6 类。
    - 先简单、稳定、好验证
    - 之后你可以扩展到 3 档波动、加入流动性、相关性等维度
    """

    trend_window: int = 20
    vol_window: int = 20
    trend_neutral_band: float = 0.0
    vol_quantile_high: float = 0.8
    vol_quantile_low: float = 0.2
    prefix: str = "regime_"


def build_regime_label_trend_vol(
    close: pd.Series,
    *,
    cfg: RegimeConfig | None = None,
) -> pd.DataFrame:
    """
    构造 regime 标签（categorical covariate）。

    直觉解释：
    - trend：过去一段时间总体是涨/跌/横盘？
    - vol：近期波动处于历史高/低分位？

    输出字段（便于人工核验）：
    - trend_score：趋势分数（动量）
    - vol_score：波动分数（滚动 std）
    - trend_cls：{-1,0,1}
    - vol_cls：{0,1}（低/高）
    - regime_code：0..5（固定编码）

    编码约定（固定，便于后续模型/回测复用）：
    - trend_cls: -1=down, 0=range, 1=up
    - vol_cls: 0=low, 1=high
    - regime_code = (trend_cls+1)*2 + vol_cls
      => down-low=0, down-high=1, range-low=2, range-high=3, up-low=4, up-high=5
    """
    if cfg is None:
        cfg = RegimeConfig()

    close = close.dropna().astype(float)
    if close.empty:
        raise ValueError("close 为空，无法构造 regime")

    idx = pd.DatetimeIndex(close.index)
    logp = pd.Series(_safe_log(close), index=idx)
    r1 = logp.diff(1)

    # 趋势：过去 trend_window 的累计 log-return
    trend = logp.diff(cfg.trend_window)

    # 波动：过去 vol_window 的波动率（log-return std）
    vol = r1.rolling(cfg.vol_window, min_periods=max(3, cfg.vol_window // 3)).std()

    # 用滚动的历史分位点作为“高/低波动”阈值（避免绝对数值阈值跨品种不可比）
    # 说明：为了人工可读，这里使用 expanding quantile（从起点开始累积）
    vol_q_high = vol.expanding(min_periods=30).quantile(cfg.vol_quantile_high)
    vol_q_low = vol.expanding(min_periods=30).quantile(cfg.vol_quantile_low)

    # trend 分类：用 0 为中性带（可设置阈值带）
    band = float(cfg.trend_neutral_band)
    trend_cls = pd.Series(np.where(trend > band, 1, np.where(trend < -band, -1, 0)), index=idx)

    # vol 分类：高/低（中间区域统一按低处理，便于先跑通；需要更细可扩展 3 档）
    vol_cls = pd.Series(np.where(vol >= vol_q_high, 1, 0), index=idx)

    regime_code = (trend_cls + 1) * 2 + vol_cls
    regime_code = regime_code.astype("Int64")  # 允许 NA（前期窗口不足）

    out = pd.DataFrame(index=idx)
    out[f"{cfg.prefix}trend_score"] = trend
    out[f"{cfg.prefix}vol_score"] = vol
    out[f"{cfg.prefix}trend_cls"] = trend_cls.astype("Int8")
    out[f"{cfg.prefix}vol_cls"] = vol_cls.astype("Int8")
    out[f"{cfg.prefix}code"] = regime_code

    return out

