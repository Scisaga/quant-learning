"""ORM 基类与元数据：统一命名规范（外键、索引、约束）。"""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# SQLAlchemy 自动生成约束/索引名时使用的模板
meta = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类。"""

    metadata = meta
