"""BacktestRun 数据访问：backtest_runs 表 CRUD。"""

from datetime import date

from sqlalchemy.orm import Session

from backend.model.backtest_run import BacktestRun


class BacktestRunRepository:
    """backtest_runs 表 CRUD。"""

    def __init__(self, session: Session):
        self._session = session

    def create(
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
    ) -> BacktestRun:
        """插入 backtest_runs 行，flush 后返回 ORM 实例。"""
        obj = BacktestRun(
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
        self._session.add(obj)
        self._session.flush()
        return obj

    def get_latest_by_recorder_id(self, recorder_id: str) -> BacktestRun | None:
        """按 recorder_id 查询最新一条。"""
        return (
            self._session.query(BacktestRun)
            .where(BacktestRun.recorder_id == recorder_id)
            .order_by(BacktestRun.created_at.desc())
            .first()
        )
