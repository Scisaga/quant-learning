from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import (
    RegimeConfig,
    build_calendar_known_future_features,
    build_past_only_features_from_close,
    build_regime_label_trend_vol,
)


@dataclass(frozen=True)
class ContextBuildConfig:
    """
    构造 `context_df` 的最小配置。

    说明：
    - Chronos-2 推理一次需要“每个 item 的历史序列”（target + past-only covariates）
    - 这里提供一个“从 close 序列出发”的最小实现，便于快速跑通
    """

    context_length: int = 200
    feature_windows: tuple[int, ...] = (5, 10, 20, 60)


def build_context_df_from_close_panel(
    close_panel: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    cfg: ContextBuildConfig | None = None,
    include_regime: bool = True,
) -> pd.DataFrame:
    """
    从 close 面板构造 Chronos-2 所需的 `context_df`。

    输入：
    - close_panel：index 为 datetime，columns 为 instrument（或 item_id），值为 close 价格
      这种“宽表”非常便于人工验证：你可以 print 一小段看每列是否对齐。
    - as_of：截断到某个时刻（不传则用最后一个时间）
    - cfg：窗口长度等配置

    输出：
    - context_df：列包含 item_id / timestamp / target + 一组 past-only covariates
      其中 target=log(close)
    """
    if cfg is None:
        cfg = ContextBuildConfig()

    if close_panel.empty:
        raise ValueError("close_panel 为空")

    close_panel = close_panel.copy()
    close_panel.index = pd.to_datetime(close_panel.index)
    close_panel = close_panel.sort_index()

    if as_of is None:
        as_of = pd.Timestamp(close_panel.index.max())
    else:
        as_of = pd.Timestamp(as_of)

    close_panel = close_panel.loc[close_panel.index <= as_of]
    if close_panel.shape[0] < cfg.context_length:
        raise ValueError(f"历史长度不足：需要至少 {cfg.context_length} 行，当前 {close_panel.shape[0]}")

    close_panel = close_panel.tail(cfg.context_length)

    rows: list[dict[str, object]] = []
    for item_id in close_panel.columns.astype(str):
        close = close_panel[item_id].dropna()
        if close.empty:
            continue

        # target & past-only 特征
        feats = build_past_only_features_from_close(close, windows=cfg.feature_windows, prefix="cov_")

        if include_regime:
            regime = build_regime_label_trend_vol(close, cfg=RegimeConfig(prefix="cov_regime_"))
            feats = feats.join(regime, how="left")

        # target：log(close)
        target = np.log(close.astype(float)).rename("target")
        feats = feats.join(target, how="left")

        for ts, row in feats.iterrows():
            # row 是一堆 cov_... + target
            rec: dict[str, object] = {"item_id": item_id, "timestamp": pd.Timestamp(ts)}
            for k, v in row.items():
                if pd.isna(v):
                    rec[k] = None
                # 注意：categorical covariate（如 regime_code）通常用整数编码表示，保留 int 便于人工核验
                elif isinstance(v, (np.integer, int)):
                    rec[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    rec[k] = float(v)
                else:
                    rec[k] = v
            rows.append(rec)

    context_df = pd.DataFrame(rows)
    if context_df.empty:
        raise ValueError("构造出的 context_df 为空（可能所有列都缺失）")

    return context_df


def build_future_df_calendar(
    item_ids: list[str],
    future_timestamps: pd.DatetimeIndex,
    *,
    prefix: str = "cov_future_",
) -> pd.DataFrame:
    """
    构造 `future_df`（known-future 协变量表）：只用日历特征。

    注意：
    - 这里的 future_timestamps 需要你自己提供“未来要预测的那些交易日”
    - 若你没有交易日历，可先用 pandas 的 BusinessDay 近似（仅用于 PoC）
    """
    cal = build_calendar_known_future_features(future_timestamps, prefix=prefix)
    rows: list[pd.DataFrame] = []
    for item in item_ids:
        df = cal.copy()
        df.insert(0, "timestamp", df.index)
        df.insert(0, "item_id", str(item))
        rows.append(df.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)
