"""配置：从环境变量加载 MinIO、PostgreSQL、持久化开关。

配置来源：项目根目录 .env，可通过环境变量覆盖。
"""

import os

from dotenv import load_dotenv

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ENV_PATH = os.path.join(_REPO_ROOT, ".env")
load_dotenv(_ENV_PATH)


def _bool_env(name: str, default: bool = False) -> bool:
    """解析环境变量为布尔值，支持 1/0、true/false、yes/no 等。"""
    v = os.getenv(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


# MinIO 配置
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "quant-experiments")
MINIO_USE_SSL = _bool_env("MINIO_USE_SSL", False)
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")

# PostgreSQL 配置：优先 DATABASE_URL，否则从分项拼接
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    _host = os.getenv("POSTGRES_HOST", "localhost")
    _port = os.getenv("POSTGRES_PORT", "5432")
    _user = os.getenv("POSTGRES_USER", "postgres")
    _password = os.getenv("POSTGRES_PASSWORD", "")
    _db = os.getenv("POSTGRES_DB", "quant_experiments")
    _password_part = f":{_password}" if _password else ""
    DATABASE_URL = f"postgresql://{_user}{_password_part}@{_host}:{_port}/{_db}"

# 是否启用持久化（未配置 MinIO/DB 时自动视为未启用）
PERSIST_EXPERIMENTS = _bool_env("PERSIST_EXPERIMENTS", False)


def is_minio_configured() -> bool:
    """MinIO 是否已配置（endpoint + 凭证）。"""
    return bool(MINIO_ENDPOINT and MINIO_ACCESS_KEY and MINIO_SECRET_KEY)


def is_db_configured() -> bool:
    """PostgreSQL 是否已配置。"""
    return bool(DATABASE_URL)
