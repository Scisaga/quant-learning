"""Backend 包：API、Model、Repository、Service、Config 等持久化与查询相关逻辑。"""

from backend.config import (  # noqa: F401 - re-export for convenience
    DATABASE_URL,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_REGION,
    MINIO_SECRET_KEY,
    MINIO_USE_SSL,
    PERSIST_EXPERIMENTS,
    is_db_configured,
    is_minio_configured,
)
from backend.repository import DbAdapter, get_session  # noqa: F401
from backend.service import (  # noqa: F401
    MinIOClient,
    persist_backtest_run,
    persist_grid_job,
    persist_grid_run_finish,
    persist_grid_run_start,
    persist_train_run,
)

__all__ = [
    "DATABASE_URL",
    "DbAdapter",
    "get_session",
    "MinIOClient",
    "MINIO_ACCESS_KEY",
    "MINIO_BUCKET",
    "MINIO_ENDPOINT",
    "MINIO_REGION",
    "MINIO_SECRET_KEY",
    "MINIO_USE_SSL",
    "PERSIST_EXPERIMENTS",
    "is_db_configured",
    "is_minio_configured",
    "persist_train_run",
    "persist_backtest_run",
    "persist_grid_run_start",
    "persist_grid_run_finish",
    "persist_grid_job",
]
