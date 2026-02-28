"""预签名 URL（Presign）API 的 Pydantic 模型。"""

from pydantic import BaseModel, Field


class PresignRequest(BaseModel):
    """预签名 URL 请求体。"""

    object_key: str = Field(..., description="MinIO 对象键（桶内完整路径）")
    expires_in: int = Field(
        3600,
        ge=60,
        le=86400,
        description="URL 有效期（秒），默认 1 小时，最大 24 小时",
    )


class PresignResponse(BaseModel):
    """预签名 URL 响应。"""

    url: str = Field(..., description="预签名 URL，用于 GET/下载")
    object_key: str = Field(..., description="请求的对象键")
    expires_in: int = Field(..., description="URL 有效期（秒）")
