"""网格批次（Grid Run）API 的 Pydantic 模型。"""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class GridRunQuery(BaseModel):
    """网格批次列表查询参数。"""

    market: str | None = Field(None, description="按市场筛选")
    status: str | None = Field(None, description="按状态筛选：running / completed / partial")
    start_date: date | None = Field(None, description="筛选：网格开始日期不早于此日期")
    end_date: date | None = Field(None, description="筛选：网格开始日期不晚于此日期")
    page: int = Field(1, ge=1, description="页码（从 1 开始）")
    page_size: int = Field(20, ge=1, le=100, description="每页条数")


class GridRunList(BaseModel):
    """网格批次列表项。"""

    id: int = Field(..., description="主键 ID")
    markets: list[str] | None = Field(None, description="市场列表")
    label_horizons: list[int] | None = Field(None, description="label horizon 列表")
    pit_grid: str | None = Field(None, description="PIT 配置：none / all / single / all+single")
    start_date: date | None = Field(None, description="网格起始日期")
    end_date: date | None = Field(None, description="网格结束日期")
    train_years: int | None = Field(None, description="训练年数")
    valid_years: int | None = Field(None, description="验证年数")
    test_years: int | None = Field(None, description="测试年数")
    step_years: int | None = Field(None, description="步长年数")
    total_jobs: int | None = Field(None, description="总 job 数")
    ok_jobs: int | None = Field(None, description="成功 job 数")
    failed_jobs: int | None = Field(None, description="失败 job 数")
    status: str | None = Field(None, description="状态")
    started_at: datetime | None = Field(None, description="开始时间")
    finished_at: datetime | None = Field(None, description="结束时间")
    created_at: datetime | None = Field(None, description="创建时间")

    model_config = {"from_attributes": True}


class GridRunDetail(GridRunList):
    """网格批次详情，含 summary/results 路径及 job 列表。"""

    minio_summary_path: str | None = Field(None, description="MinIO 上 summary.json 路径")
    minio_results_path: str | None = Field(None, description="MinIO 上 results.jsonl 路径")
    job_ids: list[int] | None = Field(None, description="归属此批次的 job ID 列表")
