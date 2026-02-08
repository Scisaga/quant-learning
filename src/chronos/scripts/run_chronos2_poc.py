"""
Chronos-2 最小 PoC 脚本（示例骨架）。

目标：
- 给定一批股票的历史窗口（context），调用 Chronos-2 一次性预测多步 ahead
- 抽取第 N 步预测，计算区间/阈值概率/score

重要说明：
1) 本脚本默认不包含“数据拉取”细节（你的数据源可能是 Qlib/CSV/数据库）。
   你需要把 `context_df` / `future_df` 组织成 Chronos-2 接口需要的形态。

2) Chronos-2 推理依赖：
   pip install "chronos-forecasting>=2.0"

运行：
  python src/chronos/scripts/run_chronos2_poc.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 让 `src/chronos/qchronos` 可被直接 import（不要求安装成 site-packages）
_ROOT = Path(__file__).resolve().parents[1]  # .../src/chronos
sys.path.insert(0, str(_ROOT))

from qchronos.chronos2_infer import Chronos2Predictor, select_n_step_forecast
from qchronos.postprocess import (
    derive_interval_and_scores,
    event_prob_from_quantiles,
    log_return_down_threshold_from_simple_return,
    log_return_threshold_from_simple_return,
)


def _make_toy_context_df() -> pd.DataFrame:
    """
    构造一个“可跑通接口”的 toy 数据（用于演示 schema）。

    真实落地时，你应当：
    - target 用 log(price)（见 docs/chronos-2.md）
    - covariates 加入板块/宽基、形态学、regime 等（past-only）
    - known-future（未来已知）放进 future_df
    """
    rng = np.random.default_rng(0)
    items = ["000001.SZ", "000002.SZ"]
    ts = pd.date_range("2020-01-01", periods=200, freq="B")  # 交易日示意（BusinessDay）

    rows = []
    for item in items:
        price = 10 + np.cumsum(rng.normal(0.0, 0.1, size=len(ts)))
        price = np.maximum(price, 0.5)
        logp = np.log(price)
        for t, y in zip(ts, logp, strict=True):
            rows.append(
                dict(
                    item_id=item,
                    timestamp=t,
                    target=float(y),
                    # past-only covariate 示例（随便放一个，便于你看到 schema）
                    cov_dummy=float(rng.normal()),
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    # ====== 0) 你需要准备的数据：context_df / future_df ======
    context_df = _make_toy_context_df()
    future_df = None  # 如有 known-future 协变量，传入对应 DataFrame

    # ====== 1) Chronos-2 推理 ======
    # 说明：model_id 以你实际选择为准（不同尺寸/版本）
    model_id = "amazon/chronos-2-small"
    predictor = Chronos2Predictor.from_pretrained(model_id)

    prediction_length = 20
    quantile_levels = [0.1, 0.5, 0.9]  # 想做阈值概率，建议换成 0.01..0.99

    forecast_df = predictor.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )

    # ====== 2) 抽取第 N 步预测（未来 N 个交易日） ======
    N = 10
    step_df = select_n_step_forecast(forecast_df, n=N)

    # ====== 3) 后处理：区间 + 阈值概率 + score ======
    # 注意：不同版本 Chronos-2 可能把分位数列命名为 '0.1'/'0.5'/'0.9' 或 'q0.1' 等
    # 这里假设列名就是 '0.1'/'0.5'/'0.9'，实际请你打印 forecast_df.columns 后对齐。
    q10_col, q50_col, q90_col = "0.1", "0.5", "0.9"

    # 事件阈值 a（简单收益），例如 3%
    a = 0.03
    u = log_return_threshold_from_simple_return(a)
    d = log_return_down_threshold_from_simple_return(a)

    # 计算阈值概率：这里需要“密集分位数网格”才更准；仅用 0.1/0.5/0.9 会很粗糙
    # 为了示例可运行，这里仍然演示接口；真实使用请把 quantile_levels 换成 0.01..0.99
    step_df["p_up_raw"] = [
        event_prob_from_quantiles(
            quantile_levels=quantile_levels,
            quantile_values=[row[q10_col], row[q50_col], row[q90_col]],
            threshold=u,
            side="up",
        )
        for _, row in step_df.iterrows()
    ]
    step_df["p_down_raw"] = [
        event_prob_from_quantiles(
            quantile_levels=quantile_levels,
            quantile_values=[row[q10_col], row[q50_col], row[q90_col]],
            threshold=d,
            side="down",
        )
        for _, row in step_df.iterrows()
    ]

    out = derive_interval_and_scores(
        step_df,
        q10_col=q10_col,
        q50_col=q50_col,
        q90_col=q90_col,
        p_up_col="p_up_raw",
        p_down_col="p_down_raw",
    )

    print("\n=== 输出字段预览（每个 item 一行：未来 N 步预测） ===")
    show_cols = [q10_col, q50_col, q90_col, "interval_width", "p_up_raw", "p_down_raw", "score1_med", "score2_med_over_width", "score3_pdiff"]
    print(out[show_cols].head(10))


if __name__ == "__main__":
    main()

