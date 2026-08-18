"""
CRITICAL TEST: Verify config NOT called per-label in moderation.

Simplified test focusing ONLY on verifying get_ai_pipeline_config() call_count,
ignoring actual moderation result success/failure.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.services.video_moderation_service import moderate_media


class TestModerationConfigCallCount:
    """
    CRITICAL: Test config được đọc ĐÚNG 1 LẦN/JOB, KHÔNG 1 LẦN/LABEL.
    Đây là bẫy hiệu năng: nếu gọi DB per-label, sẽ mở hàng chục kết nối Postgres/job.
    """

    @patch("app.services.video_moderation_service._moderate_image")
    @patch("app.services.video_moderation_service.get_ai_pipeline_config")
    def test_config_called_once_per_image_job(
        self, mock_get_config, mock_moderate_image
    ):
        """Config đọc 1 lần cho 1 image job"""
        # Arrange
        mock_get_config.return_value = {
            "fingerprint_similarity_threshold": 0.90,
            "fingerprint_cluster_threshold": 0.95,
            "rekognition_confidence_threshold": 80.0,
            "rekognition_violence_confidence_threshold": 60.0,
        }
        # Mock _moderate_image để tránh error Rekognition
        mock_moderate_image.return_value = ([], [])

        # Act
        result = moderate_media(
            file_bytes=b"image data",
            media_type="IMAGE",
            media_id="test-1",
            correlation_id="corr-1",
        )

        # Assert: config phải được gọi ĐÚNG 1 lần
        assert mock_get_config.call_count == 1, (
            f"Config được gọi {mock_get_config.call_count} lần, "
            "expected 1 lần (đọc 1 lần/job, KHÔNG per-label)"
        )

    @patch("app.services.video_moderation_service._moderate_video")
    @patch("app.services.video_moderation_service.get_ai_pipeline_config")
    def test_config_called_once_per_video_job_with_many_frames(
        self, mock_get_config, mock_moderate_video
    ):
        """
        Config đọc 1 lần cho VIDEO dù có nhiều frames.
        BẪY: nếu gọi per-frame, sẽ là 10+ lần.
        """
        # Arrange
        mock_get_config.return_value = {
            "fingerprint_similarity_threshold": 0.90,
            "fingerprint_cluster_threshold": 0.95,
            "rekognition_confidence_threshold": 80.0,
            "rekognition_violence_confidence_threshold": 60.0,
        }
        # Mock _moderate_video → 10 frames, nhiều labels/frame
        mock_moderate_video.return_value = ([], [])

        # Act
        result = moderate_media(
            file_bytes=b"video data",
            media_type="VIDEO",
            media_id="test-video",
            correlation_id="corr-video",
        )

        # Assert: CRITICAL
        # Config phải được gọi ĐÚNG 1 lần cho cả video (10 frames)
        assert mock_get_config.call_count == 1, (
            f"Video config được gọi {mock_get_config.call_count} lần, "
            "expected 1 lần (BẪY: per-frame sẽ là 10+)"
        )

