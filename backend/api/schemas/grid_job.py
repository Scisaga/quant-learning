"""网格 Job API 的 Pydantic 模型。"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class GridJobList(BaseModel):
    """网格 Job 列表项。"""

    id: int = Field(..., description="主键 ID")
    grid_run_id: int = Field(..., description="所属网格批次 ID")
    job_key: str | None = Field(None, description="job 稳定标识（与 results.jsonl 一致）")
    market: str | None = Field(None, description="市场")
    benchmark: str | None = Field(None, description="基准指数")
    label_horizon: int | None = Field(None, description="label 预测 horizon")
    pit: str | None = Field(None, description="PIT 配置名")
    status: str | None = Field(None, description="状态：ok / train_failed / report_failed")
    recorder_id: str | None = Field(None, description="Qlib recorder ID")
    train_run_id: int | None = Field(None, description="关联训练记录 ID")
    backtest_run_id: int | None = Field(None, description="关联回测记录 ID")
    started_at: datetime | None = Field(None, description="开始时间")
    finished_at: datetime | None = Field(None, description="结束时间")

    model_config = {"from_attributes": True}


class GridJobDetail(GridJobList):
    """网格 Job 详情，含配置、指标及预签名 URL。"""

    label_expr: str | None = Field(None, description="标签表达式")
    window: dict[str, Any] | None = Field(None, description="train/valid/test 时间窗口")
    minio_report_html: str | None = Field(None, description="HTML 报告 MinIO 路径")
    minio_train_log: str | None = Field(None, description="训练日志 MinIO 路径")
    minio_report_log: str | None = Field(None, description="回测日志 MinIO 路径")
    error: str | None = Field(None, description="错误信息")
    metrics: dict[str, Any] | None = Field(
        None,
        description="回测指标（IC/IR/年化收益/最大回撤等）",
    )
    params: dict[str, Any] | None = Field(
        None,
        description="从 HTML 报告提取的参数字段",
    )
    report_html_url: str | None = Field(None, description="HTML 报告预签名 URL")
    train_log_url: str | None = Field(None, description="训练日志预签名 URL")
    report_log_url: str | None = Field(None, description="回测日志预签名 URL")
