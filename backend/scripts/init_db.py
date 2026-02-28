#!/usr/bin/env python3
"""创建数据库并执行迁移（用于数据库被删除后重新初始化）。

用法:
    python -m backend.scripts.init_db [--host 10.0.0.16]
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

# 确保项目根在 path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _url_to_postgres(url: str) -> str:
    """将 DATABASE_URL 中的数据库名改为 postgres，用于连接管理库。"""
    p = urlparse(url)
    new_path = "/postgres"
    return urlunparse((p.scheme, p.netloc, new_path, p.params, p.query, p.fragment))


def create_database() -> bool:
    """创建 quant_experiments 数据库。"""
    from backend.config import DATABASE_URL

    if not DATABASE_URL:
        print("[FAIL] 未配置 DATABASE_URL")
        return False
    p = urlparse(DATABASE_URL)
    db_name = p.path.strip("/") or "quant_experiments"
    postgres_url = _url_to_postgres(DATABASE_URL)
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(postgres_url, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))
        print(f"[OK] 数据库 {db_name} 创建成功")
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"[OK] 数据库 {db_name} 已存在")
            return True
        print(f"[FAIL] 创建数据库失败: {e}")
        return False


def run_migrations() -> bool:
    """执行 backend/migrations/0001_init_schema.sql。"""
    from backend.config import DATABASE_URL
    from backend.repository.base import get_session
    from sqlalchemy import text

    if not DATABASE_URL:
        print("[FAIL] 未配置 DATABASE_URL")
        return False
    migrations_dir = Path(__file__).resolve().parent.parent / "migrations"
    sql_path = migrations_dir / "0001_init_schema.sql"
    if not sql_path.exists():
        print(f"[FAIL] 迁移文件不存在: {sql_path}")
        return False
    try:
        sql = sql_path.read_text(encoding="utf-8")
        with get_session() as session:
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if not stmt:
                    continue
                stmt = "\n".join(
                    line for line in stmt.splitlines() if not line.strip().startswith("--")
                ).strip()
                if stmt:
                    session.execute(text(stmt))
        print("[OK] 迁移执行成功")
        return True
    except Exception as e:
        print(f"[FAIL] 迁移失败: {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="创建数据库并执行迁移")
    p.add_argument("--host", default="10.0.0.16", help="PostgreSQL 地址")
    args = p.parse_args()
    host = args.host

    import os

    if host not in ("localhost", "127.0.0.1"):
        if not os.getenv("DATABASE_URL"):
            os.environ.setdefault("POSTGRES_HOST", host)
            os.environ.setdefault("POSTGRES_USER", "quant")
            os.environ.setdefault("POSTGRES_PASSWORD", "quant")
            os.environ.setdefault("POSTGRES_DB", "quant_experiments")
            os.environ.setdefault("POSTGRES_PORT", "5432")

    if not create_database():
        sys.exit(1)
    if not run_migrations():
        sys.exit(1)
    print("\n数据库初始化完成")


if __name__ == "__main__":
    main()
