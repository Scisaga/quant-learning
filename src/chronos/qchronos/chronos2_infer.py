from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pandas as pd


def _import_chronos2_pipeline() -> Any:
    """
    可选依赖导入：Chronos-2 推理管线。

    说明：
    - Amazon 的 `chronos-forecasting` 生态在不同版本里模块路径可能略有变化
    - 这里用“多路径尝试 + 清晰报错”的方式，避免让用户被 ImportError 卡住
    """
    errors: list[str] = []

    # 最常见（官方 README/示例常用）
    try:
        from chronos import Chronos2Pipeline  # type: ignore

        return Chronos2Pipeline
    except Exception as e:  # noqa: BLE001
        errors.append(f"from chronos import Chronos2Pipeline 失败：{e}")

    # 备用：有些包会用 chronos_forecasting 作为入口
    try:
        from chronos_forecasting import Chronos2Pipeline  # type: ignore

        return Chronos2Pipeline
    except Exception as e:  # noqa: BLE001
        errors.append(f"from chronos_forecasting import Chronos2Pipeline 失败：{e}")

    msg = (
        "未检测到 Chronos-2 推理依赖。\n"
        "你需要先安装 chronos-forecasting：\n"
        "  pip install \"chronos-forecasting>=2.0\"\n"
        "随后再运行推理脚本。\n"
        "导入尝试记录：\n- " + "\n- ".join(errors)
    )
    raise ImportError(msg)


@dataclass(frozen=True)
class Chronos2PredictConfig:
    """
    Chronos-2 推理配置（最小集合）。

    注意：
    - 这里的参数名尽量贴近 Chronos-2 常用接口（predict_df）
    - 具体可用参数以你安装的 `chronos-forecasting` 版本为准
    """

    prediction_length: int
    quantile_levels: Sequence[float]


class Chronos2Predictor:
    """
    Chronos-2 推理包装器（尽量少做“魔法”）。

    输入约定（与 docs/chronos-2.md 保持一致）：
    - context_df：历史可得信息
      必备列：item_id, timestamp, target
      其他列：past-only covariates（数值/类别均可）
    - future_df：未来已知信息（可为空）
      必备列：item_id, timestamp
      其他列：known-future covariates

    输出：
    - forecast_df：Chronos-2 输出（分位数预测）
      由于不同版本返回 schema 可能不同，这里不强行重塑形状。
      你可以用 `select_n_step_forecast` 抽取第 N 步的分位数行，进入后处理。
    """

    def __init__(self, pipeline: Any):
        self._pipeline = pipeline

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "Chronos2Predictor":
        Chronos2Pipeline = _import_chronos2_pipeline()
        pipe = Chronos2Pipeline.from_pretrained(model_id, **kwargs)
        return cls(pipe)

    def predict_df(
        self,
        context_df: pd.DataFrame,
        *,
        future_df: pd.DataFrame | None = None,
        prediction_length: int,
        quantile_levels: Sequence[float],
        **kwargs: Any,
    ) -> pd.DataFrame:
        _validate_context_df(context_df)
        if future_df is not None:
            _validate_future_df(future_df)

        # 说明：
        # - 我们不在这里改列名/改索引，避免猜测第三方包具体输出格式
        # - 如果你希望输出统一到某个 schema，建议在你的项目里做一层显式 adapter
        return self._pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=int(prediction_length),
            quantile_levels=list(quantile_levels),
            **kwargs,
        )


def _validate_context_df(context_df: pd.DataFrame) -> None:
    required = {"item_id", "timestamp", "target"}
    missing = required - set(context_df.columns)
    if missing:
        raise ValueError(f"context_df 缺少必备列: {sorted(missing)}")


def _validate_future_df(future_df: pd.DataFrame) -> None:
    required = {"item_id", "timestamp"}
    missing = required - set(future_df.columns)
    if missing:
        raise ValueError(f"future_df 缺少必备列: {sorted(missing)}")


def select_n_step_forecast(
    forecast_df: pd.DataFrame,
    *,
    n: int,
    item_col: str = "item_id",
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """
    从 Chronos-2 的 `forecast_df` 中抽取“第 N 步预测”（用于构造 N 日区间/概率/score）。

    为什么要这个函数？
    - Chronos-2 通常一次输出多步 ahead（t+1 .. t+K）
    - 你在策略/评测里常会固定一个 N（例如 5/10）
    - 这里把“挑第 N 步”做成显式操作，便于人工核对

    预期输入 schema（常见形态）：
    - 有列 `item_id`/`timestamp`
    - `timestamp` 是未来预测时刻（已是具体日期，而不是 horizon=1..K）

    输出：
    - 每个 item 一行（第 n 个时间点），保留原本的分位数列（例如 '0.1','0.5','0.9'）
    """
    if int(n) <= 0:
        raise ValueError("n 必须为正整数")
    n = int(n)

    if item_col not in forecast_df.columns or ts_col not in forecast_df.columns:
        raise ValueError(f"forecast_df 需要包含列 {item_col}/{ts_col}，当前列：{list(forecast_df.columns)[:20]}")

    df = forecast_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])

    # 每个 item 按 timestamp 排序后取第 n 行（1-based）
    df = df.sort_values([item_col, ts_col])
    picked = (
        df.groupby(item_col, sort=False, as_index=False)
        .nth(n - 1)
        .reset_index(drop=True)
        .set_index(item_col)
    )
    return picked

