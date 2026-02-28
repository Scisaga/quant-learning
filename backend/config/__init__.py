"""配置：从环境变量加载 MinIO、PostgreSQL、持久化开关。"""

from backend.config.config import (
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

__all__ = [
    "DATABASE_URL",
    "MINIO_ACCESS_KEY",
    "MINIO_BUCKET",
    "MINIO_ENDPOINT",
    "MINIO_REGION",
    "MINIO_SECRET_KEY",
    "MINIO_USE_SSL",
    "PERSIST_EXPERIMENTS",
    "is_db_configured",
    "is_minio_configured",
]
