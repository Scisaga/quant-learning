"""TrainRun ORM 模型，对应 train_runs 表。"""

from datetime import date, datetime
from typing import Any

from sqlalchemy import BigInteger, Date, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from backend.model.base import Base


class TrainRun(Base):
    """train_runs 表 ORM 模型。"""

    __tablename__ = "train_runs"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    recorder_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, comment="Recorder 唯一标识"
    )
    experiment_name: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="实验名称"
    )
    market: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="市场/标的"
    )
    benchmark: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="基准"
    )
    label_expr: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="标签表达式"
    )
    pit_fields: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="PIT 字段"
    )
    pit_feature_prefix: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="PIT 特征前缀"
    )
    train_start: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="训练开始日期"
    )
    train_end: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="训练结束日期"
    )
    valid_start: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="验证开始日期"
    )
    valid_end: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="验证结束日期"
    )
    test_start: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="测试开始日期"
    )
    test_end: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="测试结束日期"
    )
    handler_start: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="Handler 开始日期"
    )
    handler_end: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="Handler 结束日期"
    )
    model_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="模型配置 JSON"
    )
    minio_model_path: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="MinIO 模型存储路径"
    )
    status: Mapped[str] = mapped_column(
        String(32), default="completed", comment="状态"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间"
    )
