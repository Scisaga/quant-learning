"""BacktestRun ORM 模型，对应 backtest_runs 表。"""

from datetime import date, datetime
from typing import Any

from sqlalchemy import BigInteger, Date, DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from backend.model.base import Base


class BacktestRun(Base):
    """backtest_runs 表 ORM 模型。"""

    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    train_run_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="关联的 train_runs.id（逻辑外键，无 DB 约束）"
    )
    recorder_id: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="Recorder 唯一标识"
    )
    backtest_start: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="回测开始日期"
    )
    backtest_end: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="回测结束日期"
    )
    strategy_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="策略配置 JSON"
    )
    annualized_return: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="年化收益"
    )
    information_ratio: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="信息比率"
    )
    max_drawdown: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="最大回撤"
    )
    ic: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="IC"
    )
    icir: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="ICIR"
    )
    rank_ic: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Rank IC"
    )
    rank_icir: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Rank ICIR"
    )
    excess_return_without_cost: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="超额收益（未扣费）JSON"
    )
    excess_return_with_cost: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="超额收益（含费）JSON"
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
    status: Mapped[str] = mapped_column(
        String(32), default="completed", comment="状态"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间"
    )
