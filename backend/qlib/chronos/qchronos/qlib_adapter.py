from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


def make_pred_df_for_qlib(
    scores: pd.DataFrame,
    *,
    datetime: pd.Timestamp,
    instrument_col: str = "instrument",
    score_col: str = "score",
) -> pd.DataFrame:
    """
    把“某个 as-of 日期的一组横截面 score”整理成 Qlib 需要的 pred_df 形态。

    Qlib 约定：
    - index: MultiIndex(datetime, instrument)
    - columns: 至少包含 'score'

    输入 scores 支持两种形态：
    1) index 为 instrument（字符串），列包含 score_col
    2) 普通 DataFrame，包含 instrument_col 与 score_col
    """
    if score_col not in scores.columns:
        raise KeyError(f"缺少列: {score_col}")

    if instrument_col in scores.columns:
        inst = scores[instrument_col].astype(str)
        out = pd.DataFrame({score_col: scores[score_col].astype(float).to_numpy()}, index=inst)
    else:
        out = pd.DataFrame({score_col: scores[score_col].astype(float)})
        out.index = out.index.astype(str)

    dt = pd.Timestamp(datetime)
    out.index.name = "instrument"
    out = out.reset_index()
    out["datetime"] = dt
    out = out.set_index(["datetime", "instrument"]).sort_index()
    out = out.rename(columns={score_col: "score"})
    return out


def compute_label_from_close(
    close_df: pd.DataFrame,
    *,
    horizon: int,
    price_col: str = "close",
    kind: Literal["log_return", "simple_return"] = "log_return",
) -> pd.DataFrame:
    """
    从 close 序列计算未来 N 日标签（label_df）。

    输入：
    - close_df：index 为 MultiIndex(datetime, instrument)，列至少包含 price_col
    - horizon：N（未来 N 个交易日）
    - kind：
      - "log_return"：y_{t+N} - y_t
      - "simple_return"：(P_{t+N}/P_t)-1

    输出：
    - DataFrame(index=(datetime,instrument), columns=['label'])

    人工验证建议：
    - 随机抽几只股票，手算 (P_{t+N}/P_t-1) 看是否一致
    """
    if price_col not in close_df.columns:
        raise KeyError(f"close_df 缺少列: {price_col}")
    if int(horizon) <= 0:
        raise ValueError("horizon 必须为正整数")
    horizon = int(horizon)

    df = close_df[[price_col]].copy()
    df = df.sort_index()
    df[price_col] = df[price_col].astype(float)

    # 按 instrument 分组做 shift(-N)：得到未来价格
    future_price = df.groupby(level="instrument")[price_col].shift(-horizon)
    cur_price = df[price_col]

    if kind == "simple_return":
        label = future_price / cur_price - 1.0
    elif kind == "log_return":
        label = np.log(future_price) - np.log(cur_price)
    else:
        raise ValueError(f"未知 kind: {kind}")

    out = label.rename("label").to_frame()
    return out


def ensure_multiindex_datetime_instrument(df: pd.DataFrame) -> pd.DataFrame:
    """
    把索引规范为 MultiIndex(datetime, instrument)（若可能）。
    """
    if isinstance(df.index, pd.MultiIndex) and list(df.index.names)[:2] == ["datetime", "instrument"]:
        return df
    raise ValueError("DataFrame index 需要是 MultiIndex(datetime, instrument)")

