"""回测记录（Backtest Run）API 的 Pydantic 模型。"""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class BacktestRunList(BaseModel):
    """回测记录列表项。"""

    id: int = Field(..., description="主键 ID")
    train_run_id: int | None = Field(None, description="关联训练记录 ID")
    recorder_id: str | None = Field(None, description="Qlib recorder ID")
    backtest_start: date | None = Field(None, description="回测开始日期")
    backtest_end: date | None = Field(None, description="回测结束日期")
    annualized_return: float | None = Field(None, description="年化收益")
    information_ratio: float | None = Field(None, description="信息比率 IR")
    max_drawdown: float | None = Field(None, description="最大回撤")
    ic: float | None = Field(None, description="IC")
    icir: float | None = Field(None, description="ICIR")
    rank_ic: float | None = Field(None, description="Rank IC")
    rank_icir: float | None = Field(None, description="Rank ICIR")
    status: str | None = Field(None, description="状态：completed / failed")
    created_at: datetime | None = Field(None, description="创建时间")

    model_config = {"from_attributes": True}


class BacktestRunDetail(BacktestRunList):
    """回测记录详情。"""

    strategy_config: dict[str, Any] | None = Field(
        None,
        description="回测策略配置（strategy、topk、n_drop、hold_thresh 等）",
    )
    excess_return_without_cost: dict[str, Any] | None = Field(
        None,
        description="无成本超额收益指标",
    )
    excess_return_with_cost: dict[str, Any] | None = Field(
        None,
        description="含成本超额收益指标",
    )
    minio_report_html: str | None = Field(None, description="HTML 报告 MinIO 路径")
    minio_train_log: str | None = Field(None, description="训练日志 MinIO 路径")
    minio_report_log: str | None = Field(None, description="回测日志 MinIO 路径")
