"""
Qlib + Chronos-2 的端到端骨架（重点是“数据形态与接口对齐”）。

用途：
- 让你把现有 Qlib 数据源快速接到 Chronos-2 的 `context_df` / `future_df` schema
- 输出 Qlib 需要的 `pred_df` / `label_df`，从而复用 SigAnaRecord / PortAnaRecord

说明：
- 本脚本默认不会真的跑 Chronos-2（除非你安装了 chronos-forecasting 并配置好模型下载）
- 你可以先把数据组织打印出来人工检查，再决定是否跑大规模推理

运行（先确保 qlib 数据存在）：
  python src/chronos/scripts/qlib_end2end_stub.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# 兼容：Qlib 依赖 gym，而 gym 已废弃；仓库里已有同样的 patch
import gymnasium as gym  # type: ignore

sys.modules["gym"] = gym

# 让 `src/chronos/qchronos` 可被直接 import
_ROOT = Path(__file__).resolve().parents[1]  # .../src/chronos
sys.path.insert(0, str(_ROOT))

import qlib  # type: ignore
from qlib.constant import REG_CN  # type: ignore
from qlib.data import D  # type: ignore

from qchronos.pipeline import ContextBuildConfig, build_context_df_from_close_panel, build_future_df_calendar
from qchronos.qlib_adapter import compute_label_from_close, make_pred_df_for_qlib


def main() -> None:
    # ====== 0) 初始化 Qlib 数据源 ======
    # 你需要把 provider_uri 改成你的数据目录
    provider_uri = "data/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # ====== 1) 拉取 close（做成宽表 close_panel，便于构造 context_df） ======
    instruments = D.instruments("csi300")  # 也可以换成你的自定义池
    start_time, end_time = "2019-01-01", "2020-12-31"

    # 这里用最基础的 $close；你也可以同时拉成交额/换手用于 static gate
    close_mi = D.features(instruments, ["$close"], start_time, end_time, freq="day")
    close_mi = close_mi.rename(columns={"$close": "close"})

    # MultiIndex(datetime,instrument) -> 宽表（index=datetime, columns=instrument）
    close_panel = close_mi["close"].unstack("instrument").sort_index()

    # ====== 2) 构造 context_df（target + past-only covariates） ======
    as_of = pd.Timestamp("2020-08-31")  # 你要预测的“当日”（as-of）
    ctx_cfg = ContextBuildConfig(context_length=200)
    context_df = build_context_df_from_close_panel(close_panel, as_of=as_of, cfg=ctx_cfg, include_regime=True)

    print("\n=== context_df 预览（前 5 行；重点看列名与时间对齐）===")
    print(context_df.head())
    print("context_df columns:", list(context_df.columns))

    # ====== 3) 构造 future_df（known-future：这里只做日历特征） ======
    # 注意：真实交易日历建议用你的交易所日历；这里用 BusinessDay 近似仅作示例
    prediction_length = 20
    future_ts = pd.bdate_range(as_of + pd.offsets.BDay(1), periods=prediction_length, freq="B")
    item_ids = sorted(context_df["item_id"].unique().tolist())
    future_df = build_future_df_calendar(item_ids, future_ts)

    print("\n=== future_df 预览（前 5 行；重点看 item_id/timestamp 与日历列）===")
    print(future_df.head())

    # ====== 4) Chronos-2 推理（此处只留占位，避免没装依赖就报错） ======
    print(
        "\n[提示] 若要跑 Chronos-2 推理：\n"
        "  - 安装依赖：pip install \"chronos-forecasting>=2.0\"\n"
        "  - 然后在这里调用 qchronos.chronos2_infer.Chronos2Predictor.predict_df\n"
        "  - 得到 forecast_df 后，用 select_n_step_forecast 抽第 N 步，再做后处理生成 score\n"
    )

    # ====== 5) 先演示：如何构造 Qlib label_df（未来 N 日收益） ======
    # 真实使用时：pred_df 来自你跑 Chronos-2 得到的 score
    close_mi = close_mi.sort_index()
    label_df = compute_label_from_close(close_mi, horizon=5, price_col="close", kind="log_return")
    print("\n=== label_df 预览（前 5 行）===")
    print(label_df.head())

    # ====== 6) pred_df 示例（占位） ======
    # 假设你已经算出某天的横截面 score（index=instrument）
    dummy_scores = pd.DataFrame({"score": 0.0}, index=item_ids)
    pred_df = make_pred_df_for_qlib(dummy_scores, datetime=as_of)
    print("\n=== pred_df 预览（前 5 行）===")
    print(pred_df.head())

    print(
        "\n下一步建议：\n"
        "1) 把 Chronos-2 的 forecast_df -> 第 N 步 -> q10/q50/q90/p_up/p_down/score1..3 先做成一张表；\n"
        "2) 用 pred_df/label_df 跑 Qlib 的 SigAnaRecord（IC/ICIR）；\n"
        "3) 再把门控（static/dynamic）接到策略里跑 PortAnaRecord。\n"
    )


if __name__ == "__main__":
    main()

