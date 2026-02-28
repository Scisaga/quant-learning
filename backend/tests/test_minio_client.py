"""MinIOClient 单元测试（mock Minio，无需真实服务）。"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.service.minio_client import MinIOClient, _guess_content_type


@pytest.fixture
def mock_minio():
    """Mock Minio 客户端。"""
    with patch("backend.service.minio_client.Minio") as m:
        instance = MagicMock()
        instance.bucket_exists.return_value = True
        m.return_value = instance
        yield instance


@pytest.fixture
def client(mock_minio):
    """返回使用 mock Minio 的 MinIOClient。"""
    return MinIOClient(
        endpoint="http://localhost:9000",
        access_key="test",
        secret_key="test",
        bucket="test-bucket",
    )


class TestMinIOClient:
    """MinIOClient 测试。"""

    def test_init_ensures_bucket(self, mock_minio):
        """初始化时若 bucket 不存在则创建。"""
        mock_minio.bucket_exists.return_value = False
        MinIOClient("localhost:9000", "ak", "sk", bucket="new-bucket")
        mock_minio.make_bucket.assert_called_once_with("new-bucket")

    def test_init_skips_make_bucket_if_exists(self, mock_minio):
        """bucket 已存在时不调用 make_bucket。"""
        mock_minio.bucket_exists.return_value = True
        MinIOClient("localhost:9000", "ak", "sk", bucket="existing")
        mock_minio.make_bucket.assert_not_called()

    def test_upload_file_returns_full_path(self, client, mock_minio):
        """upload_file 返回 bucket/key 形式路径。"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"a":1}')
            path = f.name
        try:
            result = client.upload_file(path, "prefix/file.json")
            assert result == "test-bucket/prefix/file.json"
            mock_minio.fput_object.assert_called_once()
            call_kw = mock_minio.fput_object.call_args[1]
            assert call_kw["content_type"] == "application/json"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_upload_file_raises_for_missing_file(self, client):
        """upload_file 对不存在文件抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="local file not found"):
            client.upload_file("/nonexistent/path.json", "obj.json")

    def test_upload_bytes_returns_full_path(self, client, mock_minio):
        """upload_bytes 返回 bucket/key 形式路径。"""
        result = client.upload_bytes(b"hello", "prefix/bytes.bin")
        assert result == "test-bucket/prefix/bytes.bin"
        mock_minio.put_object.assert_called_once()

    def test_download_file_strips_bucket_prefix(self, client, mock_minio):
        """download_file 可处理带 bucket 前缀的 object_key。"""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "downloaded.log"
            client.download_file("test-bucket/foo/bar.log", out)
            mock_minio.fget_object.assert_called_once_with(
                "test-bucket", "foo/bar.log", str(out)
            )

    def test_download_file_uses_key_as_is_when_no_prefix(self, client, mock_minio):
        """download_file 对纯 key 直接使用。"""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "downloaded.log"
            client.download_file("foo/bar.log", out)
            mock_minio.fget_object.assert_called_once_with(
                "test-bucket", "foo/bar.log", str(out)
            )

    def test_list_objects_returns_keys(self, client, mock_minio):
        """list_objects 返回对象键列表。"""
        obj1 = MagicMock()
        obj1.object_name = "prefix/a.json"
        obj2 = MagicMock()
        obj2.object_name = "prefix/b.json"
        mock_minio.list_objects.return_value = [obj1, obj2]
        keys = client.list_objects("prefix")
        assert keys == ["prefix/a.json", "prefix/b.json"]

    def test_list_objects_respects_max_keys(self, client, mock_minio):
        """list_objects 遵守 max_keys。"""
        objs = [MagicMock(object_name=f"p/{i}.json") for i in range(10)]
        mock_minio.list_objects.return_value = iter(objs)
        keys = client.list_objects("p", max_keys=3)
        assert len(keys) == 3

    def test_get_presigned_url_strips_bucket_prefix(self, client, mock_minio):
        """get_presigned_url 可处理带 bucket 前缀的 key。"""
        mock_minio.presigned_get_object.return_value = "https://presigned.example/url"
        url = client.get_presigned_url("test-bucket/foo/bar.html")
        assert url == "https://presigned.example/url"
        mock_minio.presigned_get_object.assert_called_once_with(
            "test-bucket", "foo/bar.html", expires=3600
        )

    def test_get_presigned_url_custom_expires(self, client, mock_minio):
        """get_presigned_url 支持自定义有效期。"""
        mock_minio.presigned_get_object.return_value = "https://example/url"
        client.get_presigned_url("key", expires_seconds=600)
        mock_minio.presigned_get_object.assert_called_once_with(
            "test-bucket", "key", expires=600
        )


class TestGuessContentType:
    """_guess_content_type 测试。"""

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            (".json", "application/json"),
            (".jsonl", "application/x-ndjson"),
            (".html", "text/html"),
            (".log", "text/plain"),
            (".pkl", "application/octet-stream"),
            (".unknown", "application/octet-stream"),
        ],
    )
    def test_known_suffixes(self, suffix, expected):
        assert _guess_content_type(suffix) == expected
