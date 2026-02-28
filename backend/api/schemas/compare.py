"""多实验指标对比 API 的 Pydantic 模型。"""

from typing import Any

from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    """多实验指标对比请求体。"""

    backtest_run_ids: list[int] = Field(
        ...,
        description="待对比的回测记录 ID 列表",
    )
    grid_job_ids: list[int] | None = Field(
        None,
        description="可选：改用网格 job ID 列表进行对比",
    )


class CompareResponse(BaseModel):
    """多实验指标对比响应。"""

    labels: list[str] = Field(
        ...,
        description="实验标签（recorder_id 或 job_key）",
    )
    metrics: dict[str, list[float]] = Field(
        ...,
        description="指标名 -> 各实验对应的数值列表（如 ic、ir、annualized_return）",
    )
    raw_rows: list[dict[str, Any]] | None = Field(
        None,
        description="原始行数据，供前端自定义渲染（每行对应一个实验）",
    )
