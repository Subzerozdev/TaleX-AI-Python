"""Unit tests for app/aws/rekognition_client.py — response normalization.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.aws import rekognition_client


class TestDetectModerationLabels:
    def test_normalizes_single_label(self):
        mock_client = MagicMock()
        mock_client.detect_moderation_labels.return_value = {
            "ModerationLabels": [
                {"Name": "Explicit Nudity", "Confidence": 95.5, "ParentName": "Nudity"},
            ]
        }
        with patch.object(rekognition_client, "get_rekognition_client", return_value=mock_client):
            result = rekognition_client.detect_moderation_labels(b"fake-image-bytes")

        assert result == [
            {"name": "Explicit Nudity", "confidence": 95.5, "parent_name": "Nudity"},
        ]

    def test_normalizes_multiple_labels(self):
        mock_client = MagicMock()
        mock_client.detect_moderation_labels.return_value = {
            "ModerationLabels": [
                {"Name": "Violence", "Confidence": 88.0, "ParentName": None},
                {"Name": "Weapons", "Confidence": 70.2, "ParentName": "Violence"},
            ]
        }
        with patch.object(rekognition_client, "get_rekognition_client", return_value=mock_client):
            result = rekognition_client.detect_moderation_labels(b"fake-image-bytes")

        assert len(result) == 2
        assert result[0]["name"] == "Violence"
        assert result[1]["parent_name"] == "Violence"

    def test_missing_parent_name_defaults_to_empty_string(self):
        # ParentName field is missing entirely (top-level L1 category has no parent) — dùng
        # .get("ParentName", "") trong code thật, không phải None.
        mock_client = MagicMock()
        mock_client.detect_moderation_labels.return_value = {
            "ModerationLabels": [{"Name": "Suggestive", "Confidence": 60.0}]
        }
        with patch.object(rekognition_client, "get_rekognition_client", return_value=mock_client):
            result = rekognition_client.detect_moderation_labels(b"fake-image-bytes")

        assert result[0]["parent_name"] == ""

    def test_empty_moderation_labels_returns_empty_list(self):
        mock_client = MagicMock()
        mock_client.detect_moderation_labels.return_value = {"ModerationLabels": []}
        with patch.object(rekognition_client, "get_rekognition_client", return_value=mock_client):
            result = rekognition_client.detect_moderation_labels(b"fake-image-bytes")

        assert result == []

    def test_missing_moderation_labels_key_returns_empty_list(self):
        # response.get("ModerationLabels", []) — key hoàn toàn vắng mặt, không phải rỗng.
        mock_client = MagicMock()
        mock_client.detect_moderation_labels.return_value = {}
        with patch.object(rekognition_client, "get_rekognition_client", return_value=mock_client):
            result = rekognition_client.detect_moderation_labels(b"fake-image-bytes")

        assert result == []

    def test_calls_detect_moderation_labels_with_correct_params(self):
        mock_client = MagicMock()
        mock_client.detect_moderation_labels.return_value = {"ModerationLabels": []}
        image_bytes = b"some-real-looking-bytes"
        with patch.object(rekognition_client, "get_rekognition_client", return_value=mock_client):
            rekognition_client.detect_moderation_labels(image_bytes)

        mock_client.detect_moderation_labels.assert_called_once_with(
            Image={"Bytes": image_bytes}, MinConfidence=50.0
        )

    def test_raises_runtime_error_when_client_not_configured(self):
        with patch.object(rekognition_client, "get_rekognition_client", return_value=None):
            with pytest.raises(RuntimeError, match="Rekognition client not configured"):
                rekognition_client.detect_moderation_labels(b"fake-image-bytes")


class TestGetRekognitionClient:
    def test_returns_none_when_no_access_key_configured(self):
        # Lazy-init chỉ tạo client nếu có AWS_ACCESS_KEY_ID — reset module-level cache
        # trước/sau test để không rò rỉ trạng thái sang test khác.
        original_cache = rekognition_client._rek_client
        rekognition_client._rek_client = None
        try:
            with patch.object(rekognition_client.settings, "AWS_ACCESS_KEY_ID", ""):
                result = rekognition_client.get_rekognition_client()
            assert result is None
        finally:
            rekognition_client._rek_client = original_cache

    def test_lazy_init_caches_client_across_calls(self):
        original_cache = rekognition_client._rek_client
        rekognition_client._rek_client = None
        try:
            with patch.object(rekognition_client.settings, "AWS_ACCESS_KEY_ID", "fake-key"), \
                    patch.object(rekognition_client.settings, "AWS_SECRET_ACCESS_KEY", "fake-secret"), \
                    patch.object(rekognition_client.settings, "AWS_REGION", "ap-southeast-1"), \
                    patch("boto3.client") as mock_boto_client:
                mock_boto_client.return_value = MagicMock()
                first = rekognition_client.get_rekognition_client()
                second = rekognition_client.get_rekognition_client()

            assert first is second
            mock_boto_client.assert_called_once()
        finally:
            rekognition_client._rek_client = original_cache
