"""
qchronos：Chronos-2 落地工程的“自有”实现。

为什么不直接建一个顶层包名叫 `chronos`？
-----------------------------------
Amazon 的 `chronos-forecasting`（Chronos/Chronos-2 推理包）很可能也会使用 `chronos` 作为模块名。
为了避免与第三方包名冲突，这里用 `qchronos`（quant/qlib chronos）作为命名空间。

本包的定位：
- 不实现/复刻 Chronos-2 模型本体
- 重点实现：数据组织、后处理（区间/概率/score）、门控策略、校准、Qlib 对接
"""

from .features import (
    build_calendar_known_future_features,
    build_past_only_features_from_close,
    build_regime_label_trend_vol,
)
from .postprocess import (
    derive_interval_and_scores,
    event_prob_from_quantiles,
    make_monotone_increasing,
)
from .pipeline import (
    ContextBuildConfig,
    build_context_df_from_close_panel,
    build_future_df_calendar,
)

__all__ = [
    # features
    "build_calendar_known_future_features",
    "build_past_only_features_from_close",
    "build_regime_label_trend_vol",
    # postprocess
    "derive_interval_and_scores",
    "event_prob_from_quantiles",
    "make_monotone_increasing",
    # pipeline
    "ContextBuildConfig",
    "build_context_df_from_close_panel",
    "build_future_df_calendar",
]
