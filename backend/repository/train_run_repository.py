"""TrainRun 数据访问。"""

from datetime import date

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.model.train_run import TrainRun


class TrainRunRepository:
    """train_runs 表 CRUD。"""

    def __init__(self, session: Session):
        self._session = session

    def upsert(
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
    ) -> TrainRun:
        """按 recorder_id 插入或更新（ON CONFLICT DO UPDATE），返回 ORM 实例。"""
        stmt = insert(TrainRun).values(
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
        ).on_conflict_do_update(
            index_elements=["recorder_id"],
            set_=dict(
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
            ),
        ).returning(TrainRun.id)
        pk = self._session.execute(stmt).scalar_one()
        self._session.flush()
        return self._session.get(TrainRun, pk)

    def get_by_recorder_id(self, recorder_id: str) -> TrainRun | None:
        """按 recorder_id 查询。"""
        return (
            self._session.query(TrainRun)
            .where(TrainRun.recorder_id == recorder_id)
            .first()
        )
