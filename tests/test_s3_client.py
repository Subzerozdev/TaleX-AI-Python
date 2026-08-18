"""Unit tests for app/aws/s3_client.py — download/upload, MAX_FILE_SIZE cap.

download_from_s3() với max_bytes chặn NGAY tại HeadObject (metadata rẻ, không tải body) —
đây là lưới an toàn OOM cho kafka_consumer_service.py khi nhiều job lớn chạy song song.
Test kỹ đường cap này vì nó silent-fail rất nguy hiểm nếu bị phá vỡ (OOM cả service).
"""
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from app.aws import s3_client


class TestDownloadFromS3:
    def test_downloads_successfully_within_cap(self):
        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ContentLength": 100}
        mock_body = MagicMock()
        mock_body.read.return_value = b"x" * 100
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch.object(s3_client, "get_s3_client", return_value=mock_client):
            result = s3_client.download_from_s3("key.mp4", "bucket", max_bytes=1000)

        assert result == b"x" * 100
        mock_client.head_object.assert_called_once_with(Bucket="bucket", Key="key.mp4")
        mock_client.get_object.assert_called_once_with(Bucket="bucket", Key="key.mp4")

    def test_raises_when_content_length_exceeds_cap(self):
        # Chặn TRƯỚC khi tải — get_object() KHÔNG được gọi khi vượt cap.
        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ContentLength": 2000}

        with patch.object(s3_client, "get_s3_client", return_value=mock_client):
            with pytest.raises(ValueError, match="exceeds cap"):
                s3_client.download_from_s3("key.mp4", "bucket", max_bytes=1000)

        mock_client.get_object.assert_not_called()

    def test_proceeds_when_content_length_missing(self):
        # HeadObject đôi khi không trả ContentLength (edge case S3) — code chọn "proceed
        # without pre-check" thay vì fail cứng, chỉ log warning.
        mock_client = MagicMock()
        mock_client.head_object.return_value = {}
        mock_body = MagicMock()
        mock_body.read.return_value = b"data"
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch.object(s3_client, "get_s3_client", return_value=mock_client):
            result = s3_client.download_from_s3("key.mp4", "bucket", max_bytes=1000)

        assert result == b"data"

    def test_skips_head_object_when_max_bytes_none(self):
        # max_bytes=None (không truyền cap) — bỏ qua HeadObject hoàn toàn, tải thẳng.
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"data"
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch.object(s3_client, "get_s3_client", return_value=mock_client):
            result = s3_client.download_from_s3("key.mp4", "bucket", max_bytes=None)

        assert result == b"data"
        mock_client.head_object.assert_not_called()

    def test_uses_default_bucket_when_not_provided(self):
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"data"
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch.object(s3_client, "get_s3_client", return_value=mock_client), \
                patch.object(s3_client.settings, "AWS_S3_BUCKET", "default-bucket"):
            s3_client.download_from_s3("key.mp4")

        mock_client.get_object.assert_called_once_with(Bucket="default-bucket", Key="key.mp4")

    def test_raises_runtime_error_when_client_not_configured(self):
        with patch.object(s3_client, "get_s3_client", return_value=None):
            with pytest.raises(RuntimeError, match="S3 client not configured"):
                s3_client.download_from_s3("key.mp4", "bucket")

    def test_content_length_exactly_at_cap_is_allowed(self):
        # Boundary: content_length == max_bytes (không phải >) vẫn phải cho qua, không throw.
        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ContentLength": 1000}
        mock_body = MagicMock()
        mock_body.read.return_value = b"x" * 1000
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch.object(s3_client, "get_s3_client", return_value=mock_client):
            result = s3_client.download_from_s3("key.mp4", "bucket", max_bytes=1000)

        assert len(result) == 1000


class TestUploadToS3:
    def test_uploads_successfully(self):
        mock_client = MagicMock()
        with patch.object(s3_client, "get_s3_client", return_value=mock_client):
            s3_client.upload_to_s3("key.png", b"file-bytes", content_type="image/png", bucket="bucket")

        mock_client.put_object.assert_called_once_with(
            Bucket="bucket",
            Key="key.png",
            Body=b"file-bytes",
            ContentType="image/png",
            CacheControl="no-transform",
        )

    def test_uses_default_bucket_when_not_provided(self):
        mock_client = MagicMock()
        with patch.object(s3_client, "get_s3_client", return_value=mock_client), \
                patch.object(s3_client.settings, "AWS_S3_BUCKET", "default-bucket"):
            s3_client.upload_to_s3("key.png", b"file-bytes")

        assert mock_client.put_object.call_args.kwargs["Bucket"] == "default-bucket"

    def test_raises_runtime_error_when_client_not_configured(self):
        with patch.object(s3_client, "get_s3_client", return_value=None):
            with pytest.raises(RuntimeError, match="S3 client not configured"):
                s3_client.upload_to_s3("key.png", b"file-bytes")


class TestUploadDirToS3:
    def test_uploads_all_files_with_correct_content_types(self):
        mock_client = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            import os
            with open(os.path.join(tmp_dir, "playlist.m3u8"), "w") as f:
                f.write("#EXTM3U")
            with open(os.path.join(tmp_dir, "segment0.ts"), "wb") as f:
                f.write(b"tsdata")

            with patch.object(s3_client, "get_s3_client", return_value=mock_client):
                s3_client.upload_dir_to_s3(tmp_dir, "videos/ab_hls/media-1", bucket="bucket")

        assert mock_client.put_object.call_count == 2
        calls_by_key = {c.kwargs["Key"]: c.kwargs["ContentType"] for c in mock_client.put_object.call_args_list}
        assert calls_by_key["videos/ab_hls/media-1/playlist.m3u8"] == "application/vnd.apple.mpegurl"
        assert calls_by_key["videos/ab_hls/media-1/segment0.ts"] == "video/MP2T"

    def test_raises_runtime_error_when_client_not_configured(self):
        with patch.object(s3_client, "get_s3_client", return_value=None):
            with pytest.raises(RuntimeError, match="S3 client not configured"):
                s3_client.upload_dir_to_s3("/tmp/whatever", "prefix")
