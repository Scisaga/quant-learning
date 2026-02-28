"""训练记录（Train Run）API 的 Pydantic 模型。"""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class TrainRunList(BaseModel):
    """训练记录列表项。"""

    id: int = Field(..., description="主键 ID")
    recorder_id: str = Field(..., description="Qlib recorder ID")
    experiment_name: str | None = Field(None, description="实验名")
    market: str | None = Field(None, description="市场，如 csi300")
    benchmark: str | None = Field(None, description="基准，如 SH000300")
    label_expr: str | None = Field(None, description="标签表达式")
    train_start: date | None = Field(None, description="训练开始日期")
    train_end: date | None = Field(None, description="训练结束日期")
    test_start: date | None = Field(None, description="测试开始日期")
    test_end: date | None = Field(None, description="测试结束日期")
    status: str | None = Field(None, description="状态：completed / failed")
    created_at: datetime | None = Field(None, description="创建时间")

    model_config = {"from_attributes": True}


class TrainRunDetail(TrainRunList):
    """训练记录详情。"""

    pit_fields: str | None = Field(None, description="PIT 字段（JSON 或逗号分隔）")
    pit_feature_prefix: str | None = Field(None, description="PIT 特征前缀")
    valid_start: date | None = Field(None, description="验证开始日期")
    valid_end: date | None = Field(None, description="验证结束日期")
    handler_start: date | None = Field(None, description="Handler 开始日期")
    handler_end: date | None = Field(None, description="Handler 结束日期")
    model_config_snapshot: dict[str, Any] | None = Field(
        None,
        alias="model_config",
        description="模型配置快照（JSONB）",
    )
    minio_model_path: str | None = Field(None, description="模型 artifact MinIO 路径")

    model_config = {"from_attributes": True, "populate_by_name": True}
