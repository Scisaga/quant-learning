"""GridRun ORM 模型，对应 grid_runs 表。"""

from datetime import date, datetime

from sqlalchemy import BigInteger, Date, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from backend.model.base import Base


class GridRun(Base):
    """grid_runs 表 ORM 模型。"""

    __tablename__ = "grid_runs"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    markets: Mapped[list[str] | None] = mapped_column(
        PG_ARRAY(String), nullable=True, comment="市场列表"
    )
    label_horizons: Mapped[list[int] | None] = mapped_column(
        PG_ARRAY(Integer), nullable=True, comment="标签周期列表"
    )
    pit_grid: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="PIT 网格"
    )
    start_date: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="开始日期"
    )
    end_date: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="结束日期"
    )
    train_years: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="训练年数"
    )
    valid_years: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="验证年数"
    )
    test_years: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="测试年数"
    )
    step_years: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="步进年数"
    )
    minio_summary_path: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="MinIO 汇总路径"
    )
    minio_results_path: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="MinIO 结果路径"
    )
    total_jobs: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="总 job 数"
    )
    ok_jobs: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="成功 job 数"
    )
    failed_jobs: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="失败 job 数"
    )
    status: Mapped[str] = mapped_column(
        String(32), default="running", comment="状态"
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
