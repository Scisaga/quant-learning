"""GridJob ORM 模型，对应 grid_jobs 表。"""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from backend.model.base import Base


class GridJob(Base):
    """grid_jobs 表 ORM 模型。"""

    __tablename__ = "grid_jobs"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    grid_run_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联的 grid_runs.id（逻辑外键，无 DB 约束）"
    )
    job_key: Mapped[str] = mapped_column(
        String(256), nullable=False, comment="Job 唯一键"
    )
    market: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="市场/标的"
    )
    benchmark: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="基准"
    )
    label_horizon: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="标签周期"
    )
    label_expr: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="标签表达式"
    )
    pit: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="PIT"
    )
    # "window" 为 PostgreSQL 保留字，需显式指定列名
    window: Mapped[dict[str, Any] | None] = mapped_column(
        "window", JSONB, nullable=True, comment="窗口配置 JSON"
    )
    recorder_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="Recorder 唯一标识"
    )
    train_run_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="关联的 train_runs.id（逻辑外键，无 DB 约束）"
    )
    backtest_run_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="关联的 backtest_runs.id（逻辑外键，无 DB 约束）"
    )
    status: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="状态"
    )
    minio_report_html: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="MinIO 报告 HTML 路径"
    )
    minio_train_log: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="MinIO 训练日志路径"
    )
    minio_report_log: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="MinIO 报告日志路径"
    )
    error: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="错误信息"
    )
    metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="指标 JSON"
    )
    params: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="参数 JSON"
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="开始时间"
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="结束时间"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间"
    )
