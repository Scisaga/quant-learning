"""MinIO/S3 兼容客户端：上传、下载、列举、预签名 URL。"""

from pathlib import Path

from minio import Minio
from minio.error import S3Error


class MinIOClient:
    """MinIO/S3 兼容客户端，支持上传/下载/列举/预签名 URL。"""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str = "quant-experiments",
        use_ssl: bool = False,
        region: str = "us-east-1",
    ):
        """
        Args:
            endpoint: MinIO 服务地址，如 http://localhost:9000
            access_key: 访问密钥
            secret_key: 私钥
            bucket: 存储桶名称
            use_ssl: 是否使用 HTTPS
            region: S3 区域
        """
        self.bucket = bucket
        self._client = Minio(
            endpoint.replace("http://", "").replace("https://", ""),
            access_key=access_key,
            secret_key=secret_key,
            secure=use_ssl,
            region=region,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """确保 bucket 存在，不存在则创建。"""
        try:
            if not self._client.bucket_exists(self.bucket):
                self._client.make_bucket(self.bucket)
        except S3Error:
            pass  # 可能权限不足，后续操作会失败

    def upload_file(
        self,
        local_path: Path | str,
        object_key: str,
        content_type: str | None = None,
    ) -> str:
        """
        上传本地文件到 MinIO。

        Args:
            local_path: 本地文件路径
            object_key: 对象键（桶内路径）
            content_type: 可选的 MIME 类型

        Returns:
            对象完整路径（bucket/object_key），便于存入 DB
        """
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"local file not found: {path}")
        ct = content_type or _guess_content_type(path.suffix)
        self._client.fput_object(
            self.bucket,
            object_key,
            str(path),
            content_type=ct,
        )
        return f"{self.bucket}/{object_key}"

    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        content_type: str | None = None,
    ) -> str:
        """
        上传字节流到 MinIO。

        Args:
            data: 字节数据
            object_key: 对象键
            content_type: 可选的 MIME 类型

        Returns:
            对象完整路径
        """
        from io import BytesIO

        ct = content_type or "application/octet-stream"
        self._client.put_object(
            self.bucket,
            object_key,
            BytesIO(data),
            len(data),
            content_type=ct,
        )
        return f"{self.bucket}/{object_key}"

    def download_file(self, object_key: str, local_path: Path | str) -> None:
        """
        从 MinIO 下载对象到本地。

        Args:
            object_key: 对象键（不含 bucket 前缀；若传入 bucket/key，需先剥离 bucket）
            local_path: 本地保存路径
        """
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 若 object_key 含 bucket 前缀，则剥离
        key = object_key
        if "/" in object_key and object_key.startswith(self.bucket + "/"):
            key = object_key[len(self.bucket) + 1 :]
        self._client.fget_object(self.bucket, key, str(path))

    def list_objects(self, prefix: str, max_keys: int = 1000) -> list[str]:
        """
        列举指定前缀下的对象键。

        Returns:
            对象键列表（不含 bucket 前缀）
        """
        keys: list[str] = []
        try:
            objs = self._client.list_objects(self.bucket, prefix=prefix, recursive=True)
            for obj in objs:
                keys.append(obj.object_name)
                if len(keys) >= max_keys:
                    break
        except S3Error:
            pass
        return keys

    def get_presigned_url(self, object_key: str, expires_seconds: int = 3600) -> str:
        """
        生成预签名 GET URL，供前端直接访问/下载。

        Args:
            object_key: 对象键（可为 bucket/key 或仅 key）
            expires_seconds: 有效期（秒）

        Returns:
            预签名 URL
        """
        key = object_key
        if "/" in object_key and object_key.startswith(self.bucket + "/"):
            key = object_key[len(self.bucket) + 1 :]
        return self._client.presigned_get_object(
            self.bucket,
            key,
            expires=expires_seconds,
        )


def _guess_content_type(suffix: str) -> str:
    """根据后缀猜测 content-type。"""
    m = {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".html": "text/html",
        ".log": "text/plain",
        ".pkl": "application/octet-stream",
        ".pickle": "application/octet-stream",
    }
    return m.get(suffix.lower(), "application/octet-stream")
