"""DB 适配器：提供与 DBClient 兼容的接口，供 persist 等调用；内部使用 Repository。"""

from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session

from backend.model.backtest_run import BacktestRun
from backend.model.grid_job import GridJob
from backend.model.grid_run import GridRun
from backend.model.train_run import TrainRun
from backend.repository.backtest_run_repository import BacktestRunRepository
from backend.repository.grid_job_repository import GridJobRepository
from backend.repository.grid_run_repository import GridRunRepository
from backend.repository.train_run_repository import TrainRunRepository


class DbAdapter:
    """
    兼容 DBClient 的适配器，基于 ORM Repository 实现。
    供 persist 等模块使用；返回 ORM 实例，调用方使用属性访问（obj.id、obj.recorder_id 等）。
    """

    def __init__(self, session: Session):
        self._session = session
        self._train = TrainRunRepository(session)
        self._backtest = BacktestRunRepository(session)
        self._grid_run = GridRunRepository(session)
        self._grid_job = GridJobRepository(session)

    # ---------- train_runs ----------
    def insert_train_run(
        self,
        *,
        recorder_id: str,
        experiment_name: str,
        market: str,
        benchmark: str,
        label_expr: str | None = None,
        pit_fields: str | None = None,
        pit_feature_prefix: str | None = None,
        train_start: date | None = None,
        train_end: date | None = None,
        valid_start: date | None = None,
        valid_end: date | None = None,
        test_start: date | None = None,
        test_end: date | None = None,
        handler_start: date | None = None,
        handler_end: date | None = None,
        model_config: dict | None = None,
        minio_model_path: str | None = None,
        status: str = "completed",
    ) -> int | None:
        """插入或更新 train_runs，返回 id。"""
        row = self._train.upsert(
            recorder_id=recorder_id,
            experiment_name=experiment_name,
            market=market,
            benchmark=benchmark,
            label_expr=label_expr,
            pit_fields=pit_fields,
            pit_feature_prefix=pit_feature_prefix,
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
            handler_start=handler_start,
            handler_end=handler_end,
            model_config=model_config,
            minio_model_path=minio_model_path,
            status=status,
        )
        return row.id

    def get_train_run_by_recorder_id(self, recorder_id: str) -> TrainRun | None:
        """按 recorder_id 查询 train_runs。"""
        return self._train.get_by_recorder_id(recorder_id)

    # ---------- backtest_runs ----------
    def insert_backtest_run(
        self,
        *,
        train_run_id: int | None,
        recorder_id: str,
        backtest_start: date | None = None,
        backtest_end: date | None = None,
        strategy_config: dict | None = None,
        annualized_return: float | None = None,
        information_ratio: float | None = None,
        max_drawdown: float | None = None,
        ic: float | None = None,
        icir: float | None = None,
        rank_ic: float | None = None,
        rank_icir: float | None = None,
        excess_return_without_cost: dict | None = None,
        excess_return_with_cost: dict | None = None,
        minio_report_html: str | None = None,
        minio_train_log: str | None = None,
        minio_report_log: str | None = None,
        status: str = "completed",
    ) -> int | None:
        """插入 backtest_runs，返回 id。"""
        row = self._backtest.create(
            train_run_id=train_run_id,
            recorder_id=recorder_id,
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            strategy_config=strategy_config,
            annualized_return=annualized_return,
            information_ratio=information_ratio,
            max_drawdown=max_drawdown,
            ic=ic,
            icir=icir,
            rank_ic=rank_ic,
            rank_icir=rank_icir,
            excess_return_without_cost=excess_return_without_cost,
            excess_return_with_cost=excess_return_with_cost,
            minio_report_html=minio_report_html,
            minio_train_log=minio_train_log,
            minio_report_log=minio_report_log,
            status=status,
        )
        return row.id

    def get_backtest_run_by_recorder_id(self, recorder_id: str) -> BacktestRun | None:
        """按 recorder_id 查询 backtest_runs（最新一条）。"""
        return self._backtest.get_latest_by_recorder_id(recorder_id)

    # ---------- grid_runs ----------
    def insert_grid_run(
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
    ) -> int | None:
        """插入 grid_runs（status=running），返回 id。"""
        row = self._grid_run.create(
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
            started_at=started_at,
        )
        return row.id

    def update_grid_run(
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
        """更新 grid_runs。"""
        self._grid_run.update(
            grid_run_id,
            ok_jobs=ok_jobs,
            failed_jobs=failed_jobs,
            status=status,
            finished_at=finished_at,
            minio_summary_path=minio_summary_path,
            minio_results_path=minio_results_path,
        )

    def get_grid_run(self, grid_run_id: int) -> GridRun | None:
        """按 id 查询 grid_runs。"""
        return self._grid_run.get_by_id(grid_run_id)

    def list_grid_runs(
        self,
        *,
        market: str | None = None,
        status: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[GridRun], int]:
        """分页查询 grid_runs，返回 (rows, total_count)。"""
        return self._grid_run.list_paginated(
            market=market,
            status=status,
            start_date=start_date,
            end_date=end_date,
            page=page,
            page_size=page_size,
        )

    # ---------- grid_jobs ----------
    def insert_grid_job(
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
    ) -> int | None:
        """插入 grid_jobs，返回 id。"""
        row = self._grid_job.create(
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
        return row.id

    def get_grid_job(self, job_id: int) -> GridJob | None:
        """按 id 查询 grid_jobs。"""
        return self._grid_job.get_by_id(job_id)

    def list_grid_jobs(self, grid_run_id: int) -> list[GridJob]:
        """查询某 grid_run 下的所有 grid_jobs。"""
        return self._grid_job.list_by_grid_run_id(grid_run_id)
