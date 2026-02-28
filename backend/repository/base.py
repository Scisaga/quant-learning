"""Repository 基类：引擎与会话管理。"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.model.base import Base

_engine = None
_SessionLocal = None


def get_engine(database_url: str | None = None):
    """获取或创建 SQLAlchemy 引擎（单例，pool_pre_ping 保证连接可用）。"""
    global _engine
    if _engine is None:
        from backend.config import DATABASE_URL
        url = database_url or DATABASE_URL
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    """获取会话工厂。"""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(database_url)
        _SessionLocal = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionLocal


@contextmanager
def get_session(database_url: str | None = None) -> Generator[Session, None, None]:
    """会话上下文管理器：自动 commit/rollback/close，供脚本或非 FastAPI 请求上下文使用。"""
    factory = get_session_factory(database_url)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
