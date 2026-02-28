"""GridJob 数据访问：grid_jobs 表 CRUD。"""

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from backend.model.grid_job import GridJob


class GridJobRepository:
    """grid_jobs 表 CRUD。"""

    def __init__(self, session: Session):
        self._session = session

    def create(
        self,
        *,
        grid_run_id: int,
        job_key: str,
        market: str | None = None,
        benchmark: str | None = None,
        label_horizon: int | None = None,
        label_expr: str | None = None,
        pit: str | None = None,
        window: dict | None = None,
        recorder_id: str | None = None,
        train_run_id: int | None = None,
        backtest_run_id: int | None = None,
        status: str | None = None,
        minio_report_html: str | None = None,
        minio_train_log: str | None = None,
        minio_report_log: str | None = None,
        error: str | None = None,
        metrics: dict | None = None,
        params: dict | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> GridJob:
        """插入 grid_jobs 行，flush 后返回 ORM 实例。"""
        obj = GridJob(
            grid_run_id=grid_run_id,
            job_key=job_key,
            market=market,
            benchmark=benchmark,
            label_horizon=label_horizon,
            label_expr=label_expr,
            pit=pit,
            window=window,
            recorder_id=recorder_id,
            train_run_id=train_run_id,
            backtest_run_id=backtest_run_id,
            status=status,
            minio_report_html=minio_report_html,
            minio_train_log=minio_train_log,
            minio_report_log=minio_report_log,
            error=error,
            metrics=metrics,
            params=params,
            started_at=started_at,
            finished_at=finished_at,
        )
        self._session.add(obj)
        self._session.flush()
        return obj

    def get_by_id(self, job_id: int) -> GridJob | None:
        """按 id 查询。"""
        return self._session.get(GridJob, job_id)

    def list_by_grid_run_id(self, grid_run_id: int) -> list[GridJob]:
        """查询某 grid_run 下的所有 job。"""
        return (
            self._session.query(GridJob)
            .where(GridJob.grid_run_id == grid_run_id)
            .order_by(GridJob.id)
            .all()
        )
