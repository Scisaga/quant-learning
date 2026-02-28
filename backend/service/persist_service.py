"""持久化编排：训练/回测/网格产物的 MinIO 上传与 PostgreSQL 写入。

供 run_grid、train、generate_html_report 等脚本调用；
需配合 DbAdapter 与 MinIOClient 使用。
"""

from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.repository.db_adapter import DbAdapter  # noqa: PLC0415
    from backend.service.minio_client import MinIOClient  # noqa: PLC0415


def _parse_date_or_none(s: str | None) -> date | None:
    """解析日期字符串，取前 10 位（YYYY-MM-DD）。"""
    if not s:
        return None
    try:
        from datetime import datetime as dt
        return dt.strptime(s.strip()[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _parse_datetime_or_none(s: str | None) -> datetime | None:
    """解析 ISO 或通用日期时间字符串。"""
    if not s:
        return None
    try:
        from dateutil import parser as dup
        return dup.parse(s)
    except Exception:
        return None


def persist_train_run(
    *,
    recorder_id: str,
    experiment_name: str,
    market: str,
    benchmark: str,
    minio_client: "MinIOClient",
    db_client: "DbAdapter",
    label_expr: str | None = None,
    pit_fields: str | None = None,
    pit_feature_prefix: str | None = None,
    train_start: date | str | None = None,
    train_end: date | str | None = None,
    valid_start: date | str | None = None,
    valid_end: date | str | None = None,
    test_start: date | str | None = None,
    test_end: date | str | None = None,
    handler_start: date | str | None = None,
    handler_end: date | str | None = None,
    model_config: dict | None = None,
    artifact_paths: dict[str, Path | str] | None = None,
) -> int | None:
    """
    训练完成后持久化：上传 artifact 到 MinIO，写入 train_runs。

    Args:
        recorder_id: Qlib recorder ID
        experiment_name: 实验名
        market: 市场
        benchmark: 基准
        minio_client: MinIO 客户端
        db_client: DB 客户端
        artifact_paths: 本地文件路径字典，key 如 run_config, trained_model 等
    """
    from datetime import datetime as dt

    prefix = f"train/{dt.now().strftime('%Y-%m-%d')}/{experiment_name}_{recorder_id}"
    minio_model_path: str | None = None
    if artifact_paths:
        for name, local_path in artifact_paths.items():
            p = Path(local_path)
            if not p.exists():
                continue
            obj_key = f"{prefix}/{name}.pkl" if not p.suffix else f"{prefix}/{p.name}"
            minio_client.upload_file(p, obj_key)
            if name in ("trained_model", "model"):
                minio_model_path = f"{minio_client.bucket}/{obj_key}"

    def _d(v: date | str | None) -> date | None:
        if v is None:
            return None
        if isinstance(v, date) and not isinstance(v, datetime):
            return v
        return _parse_date_or_none(str(v))

    return db_client.insert_train_run(
        recorder_id=recorder_id,
        experiment_name=experiment_name,
        market=market,
        benchmark=benchmark,
        label_expr=label_expr,
        pit_fields=pit_fields,
        pit_feature_prefix=pit_feature_prefix,
        train_start=_d(train_start),
        train_end=_d(train_end),
        valid_start=_d(valid_start),
        valid_end=_d(valid_end),
        test_start=_d(test_start),
        test_end=_d(test_end),
        handler_start=_d(handler_start),
        handler_end=_d(handler_end),
        model_config=model_config,
        minio_model_path=minio_model_path,
    )


def persist_backtest_run(
    *,
    recorder_id: str,
    minio_client: "MinIOClient",
    db_client: "DbAdapter",
    experiment_name: str,
    report_html_path: Path | str | None = None,
    train_log_path: Path | str | None = None,
    report_log_path: Path | str | None = None,
    train_run_id: int | None = None,
    backtest_start: date | str | None = None,
    backtest_end: date | str | None = None,
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
) -> int | None:
    """
    回测完成后持久化：上传 report.html、train.log、report.log 到 MinIO，写入 backtest_runs。

    若 train_run_id 未提供，会尝试按 recorder_id 查询 train_runs 获取。
    """
    prefix = f"backtest/{experiment_name}_{recorder_id}"
    minio_report_html: str | None = None
    minio_train_log: str | None = None
    minio_report_log: str | None = None

    for name, local_path in (
        ("report.html", report_html_path),
        ("train.log", train_log_path),
        ("report.log", report_log_path),
    ):
        p = Path(local_path) if local_path else None
        if not p or not p.exists():
            continue
        obj_key = f"{prefix}/{p.name}"
        minio_client.upload_file(p, obj_key)
        full = f"{minio_client.bucket}/{obj_key}"
        if name == "report.html":
            minio_report_html = full
        elif name == "train.log":
            minio_train_log = full
        else:
            minio_report_log = full

    if train_run_id is None:
        tr = db_client.get_train_run_by_recorder_id(recorder_id)
        train_run_id = tr.id if tr else None

    def _d(v: date | str | None) -> date | None:
        if v is None:
            return None
        if isinstance(v, date) and not isinstance(v, datetime):
            return v
        return _parse_date_or_none(str(v))

    return db_client.insert_backtest_run(
        train_run_id=train_run_id,
        recorder_id=recorder_id,
        backtest_start=_d(backtest_start),
        backtest_end=_d(backtest_end),
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
    )


def persist_grid_run_start(
    *,
    db_client: "DbAdapter",
    summary: dict[str, Any],
) -> int | None:
    """
    Grid 开始时插入 grid_runs（status=running），返回 grid_run_id。
    """
    start_d = _parse_date_or_none(summary.get("start_date"))
    end_d = _parse_date_or_none(summary.get("end_date"))
    return db_client.insert_grid_run(
        markets=summary.get("markets"),
        label_horizons=summary.get("label_horizons"),
        pit_grid=summary.get("pit_grid") or (summary.get("pit_specs", [None])[0] if summary.get("pit_specs") else None),
        start_date=start_d,
        end_date=end_d,
        train_years=summary.get("train_years"),
        valid_years=summary.get("valid_years"),
        test_years=summary.get("test_years"),
        step_years=summary.get("step_years"),
        total_jobs=summary.get("num_jobs") or summary.get("num_todo"),
        started_at=datetime.now(),
    )


def persist_grid_run_finish(
    *,
    grid_run_id: int,
    db_client: "DbAdapter",
    minio_client: "MinIOClient",
    out_dir: Path | str,
    ok_jobs: int,
    failed_jobs: int,
) -> None:
    """
    Grid 全部完成后：上传 summary.json、results.jsonl、logs/* 到 MinIO，更新 grid_runs。
    """
    out = Path(out_dir)
    minio_summary_path: str | None = None
    minio_results_path: str | None = None

    prefix = f"grid/{grid_run_id}"
    summary_path = out / "summary.json"
    if summary_path.exists():
        minio_summary_path = minio_client.upload_file(summary_path, f"{prefix}/summary.json")

    results_path = out / "results.jsonl"
    if results_path.exists():
        minio_results_path = minio_client.upload_file(results_path, f"{prefix}/results.jsonl")

    logs_dir = out / "logs"
    if logs_dir.is_dir():
        for log_file in logs_dir.iterdir():
            if log_file.is_file():
                minio_client.upload_file(log_file, f"{prefix}/logs/{log_file.name}")

    status = "completed" if failed_jobs == 0 else "partial"
    db_client.update_grid_run(
        grid_run_id,
        ok_jobs=ok_jobs,
        failed_jobs=failed_jobs,
        status=status,
        finished_at=datetime.now(),
        minio_summary_path=minio_summary_path,
        minio_results_path=minio_results_path,
    )


def persist_grid_job(
    *,
    grid_run_id: int,
    job_key: str,
    market: str,
    benchmark: str,
    label_horizon: int,
    label_expr: str,
    pit: str,
    window: dict | None,
    status: str,
    recorder_id: str | None,
    started_at: str | None,
    finished_at: str | None,
    minio_client: "MinIOClient",
    db_client: "DbAdapter",
    experiment_name: str,
    report_html_path: Path | str | None = None,
    train_log_path: Path | str | None = None,
    report_log_path: Path | str | None = None,
    error: str | None = None,
    metrics: dict | None = None,
    params: dict | None = None,
) -> int | None:
    """
    单个 job 完成后：上传 report.html、train.log、report.log 到 MinIO，
    插入/更新 train_runs/backtest_runs（若有 recorder_id），插入 grid_jobs。
    """
    prefix = f"grid/{grid_run_id}/jobs/{job_key}"
    minio_report_html: str | None = None
    minio_train_log: str | None = None
    minio_report_log: str | None = None

    for name, local_path in (
        ("report.html", report_html_path),
        ("train.log", train_log_path),
        ("report.log", report_log_path),
    ):
        p = Path(local_path) if local_path else None
        if not p or not p.exists():
            continue
        obj_key = f"{prefix}/{p.name}"
        minio_client.upload_file(p, obj_key)
        full = f"{minio_client.bucket}/{obj_key}"
        if name == "report.html":
            minio_report_html = full
        elif name == "train.log":
            minio_train_log = full
        else:
            minio_report_log = full

    train_run_id: int | None = None
    backtest_run_id: int | None = None

    if recorder_id:
        tr = db_client.get_train_run_by_recorder_id(recorder_id)
        train_run_id = tr.id if tr else None
        br = db_client.get_backtest_run_by_recorder_id(recorder_id)
        backtest_run_id = br.id if br else None
        # 若 train/backtest 尚未持久化，这里可按需插入简化版；当前设计由 run_grid 统一在 job 完成时调用
        # persist_backtest_run，故通常 backtest_run 已存在。若未存在，可在此处补一次 persist_backtest_run。
        if not backtest_run_id and status == "ok" and (report_html_path or report_log_path):
            # metrics 来自 _extract_metrics_from_report_log，键为 flat 形式如 excess_return_without_cost.annualized_return
            m = metrics or {}
            er_without = {k.replace("excess_return_without_cost.", ""): v for k, v in m.items() if k.startswith("excess_return_without_cost.")}
            er_with = {k.replace("excess_return_with_cost.", ""): v for k, v in m.items() if k.startswith("excess_return_with_cost.")}
            backtest_run_id = persist_backtest_run(
                recorder_id=recorder_id,
                experiment_name=experiment_name,
                minio_client=minio_client,
                db_client=db_client,
                report_html_path=report_html_path,
                train_log_path=train_log_path,
                report_log_path=report_log_path,
                train_run_id=train_run_id,
                annualized_return=er_without.get("annualized_return") or er_with.get("annualized_return"),
                information_ratio=er_without.get("information_ratio") or er_with.get("information_ratio"),
                max_drawdown=er_without.get("max_drawdown") or er_with.get("max_drawdown"),
                ic=m.get("IC"),
                icir=m.get("ICIR"),
                rank_ic=m.get("RankIC"),
                rank_icir=m.get("RankICIR"),
                excess_return_without_cost=er_without if er_without else None,
                excess_return_with_cost=er_with if er_with else None,
            )

    return db_client.insert_grid_job(
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
        started_at=_parse_datetime_or_none(started_at),
        finished_at=_parse_datetime_or_none(finished_at),
    )
