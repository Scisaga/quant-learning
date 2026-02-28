"""GridRun 数据访问：grid_runs 表 CRUD（含分页、筛选）。"""

from datetime import date, datetime
from typing import Any

from sqlalchemy import and_, nulls_last
from sqlalchemy.orm import Session

from backend.model.grid_run import GridRun


class GridRunRepository:
    """grid_runs 表 CRUD。"""

    def __init__(self, session: Session):
        self._session = session

    def create(
        self,
        *,
        markets: list[str] | None = None,
        label_horizons: list[int] | None = None,
        pit_grid: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        train_years: int | None = None,
        valid_years: int | None = None,
        test_years: int | None = None,
        step_years: int | None = None,
        total_jobs: int | None = None,
        started_at: datetime | None = None,
    ) -> GridRun:
        """插入 grid_runs（status=running，ok_jobs/failed_jobs=0），返回 ORM 实例。"""
        obj = GridRun(
            markets=markets,
            label_horizons=label_horizons,
            pit_grid=pit_grid,
            start_date=start_date,
            end_date=end_date,
            train_years=train_years,
            valid_years=valid_years,
            test_years=test_years,
            step_years=step_years,
            total_jobs=total_jobs,
            ok_jobs=0,
            failed_jobs=0,
            status="running",
            started_at=started_at,
        )
        self._session.add(obj)
        self._session.flush()
        return obj

    def get_by_id(self, grid_run_id: int) -> GridRun | None:
        """按 id 查询。"""
        return self._session.get(GridRun, grid_run_id)

    def update(
        self,
        grid_run_id: int,
        *,
        ok_jobs: int | None = None,
        failed_jobs: int | None = None,
        status: str | None = None,
        finished_at: datetime | None = None,
        minio_summary_path: str | None = None,
        minio_results_path: str | None = None,
    ) -> None:
        """更新指定字段。"""
        stmt = (
            GridRun.__table__.update()
            .where(GridRun.id == grid_run_id)
        )
        updates: dict[str, Any] = {}
        if ok_jobs is not None:
            updates["ok_jobs"] = ok_jobs
        if failed_jobs is not None:
            updates["failed_jobs"] = failed_jobs
        if status is not None:
            updates["status"] = status
        if finished_at is not None:
            updates["finished_at"] = finished_at
        if minio_summary_path is not None:
            updates["minio_summary_path"] = minio_summary_path
        if minio_results_path is not None:
            updates["minio_results_path"] = minio_results_path
        if updates:
            self._session.execute(stmt.values(**updates))
            self._session.flush()

    def list_paginated(
        self,
        *,
        market: str | None = None,
        status: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[GridRun], int]:
        """分页查询 grid_runs，支持 market/status/start_date/end_date 筛选；返回 (rows, total_count)。"""
        conditions = []
        if market:
            conditions.append(GridRun.markets.contains([market]))
        if status:
            conditions.append(GridRun.status == status)
        if start_date:
            conditions.append(GridRun.start_date >= start_date)
        if end_date:
            conditions.append(GridRun.start_date <= end_date)

        q = self._session.query(GridRun)
        if conditions:
            q = q.where(and_(*conditions))

        total = q.count()
        offset = (page - 1) * page_size
        rows = (
            q.order_by(nulls_last(GridRun.started_at.desc()))
            .offset(offset)
            .limit(page_size)
            .all()
        )
        return rows, total
