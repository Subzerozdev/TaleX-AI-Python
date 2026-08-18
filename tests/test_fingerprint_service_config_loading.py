"""
Test fingerprint_service.process_fingerprint() config loading pattern.

Kiểm tra:
1. Config được đọc ĐÚNG 1 LẦN/job (não ai_config = get_ai_pipeline_config() ở đầu)
2. Config được truyền xuống resolve_content_cluster() và _find_violations()
3. KHÔNG gọi DB trong vòng lặp xử lý frames/segments
"""

import pytest
from unittest.mock import patch, MagicMock
from app.services.fingerprint_service import process_fingerprint


def _full_config(**overrides) -> dict:
    """Dict cấu hình động đủ 13 key (khớp get_ai_pipeline_config)."""
    cfg = {
        "fingerprint_similarity_threshold": 0.90,
        "fingerprint_cluster_threshold": 0.95,
        "rekognition_confidence_threshold": 80.0,
        "rekognition_violence_confidence_threshold": 60.0,
        "fingerprint_image_top_k": 20,
        "fingerprint_video_top_k": 15,
        "fingerprint_min_match_seconds": 5,
        "fingerprint_max_gap_seconds": 2,
        "fingerprint_fps": 1,
        "fingerprint_max_frames": 300,
        "fingerprint_max_file_size_mb": 100,
        "rekognition_max_frames": 30,
        "moderation_frame_interval": 2.0,
    }
    cfg.update(overrides)
    return cfg


class TestFingerprintServiceConfigLoadingPerJob:
    """
    Kiểm tra config đọc 1 lần/job trong process_fingerprint().
    """

    @patch("app.services.fingerprint_service.get_ai_pipeline_config")
    @patch("app.services.fingerprint_service.is_connected")
    @patch("app.services.fingerprint_service._validate_file")
    @patch("app.services.fingerprint_service._process_image")
    @patch("app.services.fingerprint_service.resolve_content_cluster")
    @patch("app.services.fingerprint_service.delete_by_media_id")
    @patch("app.services.fingerprint_service._find_violations")
    @patch("app.services.fingerprint_service.insert_fingerprints")
    def test_config_loaded_once_per_image_job(
        self,
        mock_insert,
        mock_find_violations,
        mock_delete,
        mock_resolve_cluster,
        mock_process_image,
        mock_validate,
        mock_is_connected,
        mock_get_config,
    ):
        """Config đọc 1 lần khi xử lý 1 image"""
        # Arrange
        mock_is_connected.return_value = True
        mock_get_config.return_value = _full_config()

        # Mock fingerprint extraction
        mock_process_image.return_value = [
            {"vector": [0.1, 0.2, 0.3], "frame_index": 0, "hash": "hash1"}
        ]

        # Mock cluster resolution
        mock_resolve_cluster.return_value = MagicMock(
            matched=False,
            original_creator_id=None,
        )

        # Mock violation finding
        mock_find_violations.return_value = []

        # Mock insert
        mock_insert.return_value = None

        # Act
        result = process_fingerprint(
            media_id="test-image-1",
            creator_id="creator-1",
            file_bytes=b"fake image data",
            filename="test.jpg",
        )

        # Assert
        # Config phải được gọi ĐÚNG 1 LẦN
        assert mock_get_config.call_count == 1, (
            f"Config được gọi {mock_get_config.call_count} lần, "
            "expected 1 lần (đầu process_fingerprint)"
        )
        assert result is not None

    @patch("app.services.fingerprint_service.get_ai_pipeline_config")
    @patch("app.services.fingerprint_service.is_connected")
    @patch("app.services.fingerprint_service._validate_file")
    @patch("app.services.fingerprint_service._process_video")
    @patch("app.services.fingerprint_service.resolve_content_cluster")
    @patch("app.services.fingerprint_service.delete_by_media_id")
    @patch("app.services.fingerprint_service._find_violations")
    @patch("app.services.fingerprint_service.insert_fingerprints")
    def test_config_loaded_once_for_video_multiple_frames(
        self,
        mock_insert,
        mock_find_violations,
        mock_delete,
        mock_resolve_cluster,
        mock_process_video,
        mock_validate,
        mock_is_connected,
        mock_get_config,
    ):
        """
        Config đọc 1 lần cho VIDEO dù có nhiều frames.
        BẪY: nếu gọi DB per-frame sẽ là 10+ lần cho 10 frames.
        """
        # Arrange
        mock_is_connected.return_value = True
        mock_get_config.return_value = _full_config(
            fingerprint_similarity_threshold=0.88,
            fingerprint_cluster_threshold=0.93,
            rekognition_confidence_threshold=78.0,
            rekognition_violence_confidence_threshold=58.0,
        )

        # Mock video processing: 10 frames
        fingerprints = [
            {"vector": [i * 0.1, i * 0.2, i * 0.3], "frame_index": i, "hash": f"hash{i}"}
            for i in range(10)
        ]
        mock_process_video.return_value = fingerprints

        # Mock cluster resolution
        mock_resolve_cluster.return_value = MagicMock(
            matched=False,
            original_creator_id=None,
        )

        # Mock violation finding
        mock_find_violations.return_value = []

        # Act
        result = process_fingerprint(
            media_id="test-video-1",
            creator_id="creator-1",
            file_bytes=b"fake video data",
            filename="test.mp4",
        )

        # Assert
        # Config PHẢI được gọi ĐÚNG 1 LẦN dù có 10 frames
        assert mock_get_config.call_count == 1, (
            f"Video config được gọi {mock_get_config.call_count} lần, "
            "expected 1 lần (BẪY: per-frame sẽ là 10 lần)"
        )
        assert result is not None

    @patch("app.services.fingerprint_service.get_ai_pipeline_config")
    @patch("app.services.fingerprint_service.is_connected")
    @patch("app.services.fingerprint_service._validate_file")
    @patch("app.services.fingerprint_service._process_image")
    @patch("app.services.fingerprint_service.resolve_content_cluster")
    @patch("app.services.fingerprint_service.delete_by_media_id")
    @patch("app.services.fingerprint_service._find_violations")
    @patch("app.services.fingerprint_service.insert_fingerprints")
    def test_config_passed_to_resolve_content_cluster(
        self,
        mock_insert,
        mock_find_violations,
        mock_delete,
        mock_resolve_cluster,
        mock_process_image,
        mock_validate,
        mock_is_connected,
        mock_get_config,
    ):
        """Config được truyền xuống resolve_content_cluster()"""
        # Arrange
        config = _full_config()
        mock_is_connected.return_value = True
        mock_get_config.return_value = config

        mock_process_image.return_value = [
            {"vector": [0.1, 0.2, 0.3], "frame_index": 0, "hash": "hash1"}
        ]

        mock_resolve_cluster.return_value = MagicMock(
            matched=False,
            original_creator_id=None,
        )

        mock_find_violations.return_value = []

        # Act
        process_fingerprint(
            media_id="test-1",
            creator_id="creator-1",
            file_bytes=b"data",
            filename="test.jpg",
        )

        # Assert
        # resolve_content_cluster phải nhận cluster_threshold từ config
        assert mock_resolve_cluster.called
        call_args = mock_resolve_cluster.call_args
        # Verify cluster_threshold được truyền
        assert call_args.kwargs.get("cluster_threshold") == 0.95

    @patch("app.services.fingerprint_service.get_ai_pipeline_config")
    @patch("app.services.fingerprint_service.is_connected")
    @patch("app.services.fingerprint_service._validate_file")
    @patch("app.services.fingerprint_service._process_image")
    @patch("app.services.fingerprint_service.resolve_content_cluster")
    @patch("app.services.fingerprint_service.delete_by_media_id")
    @patch("app.services.fingerprint_service._find_violations")
    @patch("app.services.fingerprint_service.insert_fingerprints")
    def test_config_passed_to_find_violations(
        self,
        mock_insert,
        mock_find_violations,
        mock_delete,
        mock_resolve_cluster,
        mock_process_image,
        mock_validate,
        mock_is_connected,
        mock_get_config,
    ):
        """Config được truyền xuống _find_violations()"""
        # Arrange
        config = _full_config(
            fingerprint_similarity_threshold=0.87,
            fingerprint_cluster_threshold=0.92,
        )
        mock_is_connected.return_value = True
        mock_get_config.return_value = config

        mock_process_image.return_value = [
            {"vector": [0.1, 0.2, 0.3], "frame_index": 0, "hash": "hash1"}
        ]

        mock_resolve_cluster.return_value = MagicMock(
            matched=False,
            original_creator_id=None,
        )

        mock_find_violations.return_value = []

        # Act
        process_fingerprint(
            media_id="test-2",
            creator_id="creator-1",
            file_bytes=b"data",
            filename="test.jpg",
        )

        # Assert
        # _find_violations nhận nguyên dict ai_config (positional arg thứ 2) — mọi ngưỡng
        # được đọc bên trong từ dict này, không truyền lẻ similarity_threshold nữa.
        assert mock_find_violations.called
        call_args = mock_find_violations.call_args
        assert config in call_args.args
        assert call_args.args[1]["fingerprint_similarity_threshold"] == 0.87

    @patch("app.services.fingerprint_service.get_ai_pipeline_config")
    @patch("app.services.fingerprint_service.is_connected")
    @patch("app.services.fingerprint_service._validate_file")
    @patch("app.services.fingerprint_service._process_image")
    @patch("app.services.fingerprint_service.resolve_content_cluster")
    @patch("app.services.fingerprint_service.delete_by_media_id")
    @patch("app.services.fingerprint_service._find_violations")
    @patch("app.services.fingerprint_service.insert_fingerprints")
    def test_config_not_called_after_initial_read(
        self,
        mock_insert,
        mock_find_violations,
        mock_delete,
        mock_resolve_cluster,
        mock_process_image,
        mock_validate,
        mock_is_connected,
        mock_get_config,
    ):
        """
        Config không được gọi lại sau lần đầu.
        Nếu code có gọi get_ai_pipeline_config() bên trong vòng lặp hoặc
        bên trong _find_violations(), test fail.
        """
        # Arrange
        call_count_tracker = {"count": 0}

        def track_get_config():
            call_count_tracker["count"] += 1
            return _full_config()

        mock_is_connected.return_value = True
        mock_get_config.side_effect = track_get_config

        mock_process_image.return_value = [
            {"vector": [0.1, 0.2, 0.3], "frame_index": 0, "hash": "hash1"}
        ]

        mock_resolve_cluster.return_value = MagicMock(
            matched=False,
            original_creator_id=None,
        )

        # Giả lập _find_violations gọi get_ai_pipeline_config (BẪY)
        # → nếu code thực sự có gọi, call_count sẽ > 1
        mock_find_violations.return_value = []

        # Act
        process_fingerprint(
            media_id="test-3",
            creator_id="creator-1",
            file_bytes=b"data",
            filename="test.jpg",
        )

        # Assert
        # Config được gọi ĐÚNG 1 LẦN
        assert call_count_tracker["count"] == 1, (
            f"Config được gọi {call_count_tracker['count']} lần, "
            "expected 1 lần (BẪY: nếu gọi lại trong loop sẽ > 1)"
        )
