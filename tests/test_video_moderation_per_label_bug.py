"""
CRITICAL TEST: Video Moderation Per-Label Config Loading Bug Prevention.

Phát hiện bẫy hiệu năng: nếu get_ai_pipeline_config() được gọi PER-LABEL thay vì
1 LẦN/JOB, sẽ mở hàng chục kết nối Postgres/job → lỗi hiệu năng nghiêm trọng.

Test này dùng Mock.call_count để kiểm tra get_ai_pipeline_config() được gọi ĐÚNG
1 LẦN cho 1 media job dù có nhiều label vi phạm.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from app.services.video_moderation_service import moderate_media


class TestVideoModerationConfigLoadingPerJob:
    """
    Kiểm tra config được đọc ĐÚNG 1 LẦN/JOB, KHÔNG 1 LẦN/LABEL.
    """

    @patch("app.services.video_moderation_service._normalize_for_rekognition")
    @patch("app.services.video_moderation_service.detect_moderation_labels")
    @patch("app.services.video_moderation_service.get_ai_pipeline_config")
    def test_config_loaded_once_per_media_with_single_violation(
        self, mock_get_config, mock_detect_labels, mock_normalize
    ):
        """Config đọc 1 lần khi media có 1 label vi phạm"""
        # Arrange
        mock_get_config.return_value = {
            "fingerprint_similarity_threshold": 0.90,
            "fingerprint_cluster_threshold": 0.95,
            "rekognition_confidence_threshold": 80.0,
            "rekognition_violence_confidence_threshold": 60.0,
        }

        # Mock normalize
        mock_normalize.return_value = b"normalized data"

        # Mock Rekognition: 1 frame, 1 label vi phạm — detect_moderation_labels() đã tự
        # chuẩn hóa response AWS thành list[dict] key thường (xem rekognition_client.py),
        # KHÔNG phải dict thô {"ModerationLabels": [...]} với key hoa như response AWS gốc.
        mock_detect_labels.return_value = [
            {"name": "Adult", "confidence": 85.0, "parent_name": None},
        ]

        # Act
        result = moderate_media(
            file_bytes=b"fake image data",
            media_type="IMAGE",
            media_id="test-media-1",
            correlation_id="corr-1",
        )

        # Assert
        # get_ai_pipeline_config phải được gọi ĐÚNG 1 LẦN, KHÔNG phải per-label
        assert mock_get_config.call_count == 1, (
            f"Config được gọi {mock_get_config.call_count} lần, "
            "expected 1 lần (đọc 1 lần/job, không phải per-label)"
        )
        assert result["success"] is True

    @patch("app.services.video_moderation_service._normalize_for_rekognition")
    @patch("app.services.video_moderation_service.detect_moderation_labels")
    @patch("app.services.video_moderation_service.get_ai_pipeline_config")
    def test_config_loaded_once_with_multiple_violations(
        self, mock_get_config, mock_detect_labels, mock_normalize
    ):
        """
        Config đọc 1 lần dù media có NHIỀU label vi phạm.
        ĐÂY LÀ BẪY HIỆU NĂNG: nếu code sai mở DB per-label, sẽ gọi 5+ lần.
        """
        # Arrange
        mock_normalize.return_value = b"normalized"
        mock_get_config.return_value = {
            "fingerprint_similarity_threshold": 0.90,
            "fingerprint_cluster_threshold": 0.95,
            "rekognition_confidence_threshold": 80.0,
            "rekognition_violence_confidence_threshold": 60.0,
        }

        # Mock Rekognition: 1 frame, 5 labels vi phạm (gây áp lực trên DB) — shape đã
        # chuẩn hóa (xem giải thích ở test đầu file).
        mock_detect_labels.return_value = [
            {"name": "Violence", "confidence": 88.0, "parent_name": None},
            {"name": "Visually Disturbing", "confidence": 82.0, "parent_name": None},
            {"name": "Adult", "confidence": 75.0, "parent_name": None},
            {"name": "Suggestive", "confidence": 70.0, "parent_name": "Adult"},
            {"name": "Weapons", "confidence": 65.0, "parent_name": None},
        ]

        # Act
        result = moderate_media(
            file_bytes=b"fake image with multiple violations",
            media_type="IMAGE",
            media_id="test-media-multi",
            correlation_id="corr-multi",
        )

        # Assert
        # CRITICAL: 5 labels vi phạm, nhưng config chỉ được gọi 1 lần
        assert mock_get_config.call_count == 1, (
            f"Config được gọi {mock_get_config.call_count} lần với 5 labels, "
            "expected 1 lần (BẪY: nếu gọi per-label sẽ là 5 lần)"
        )
        assert result["success"] is True
        assert len(result["violations"]) > 0

    @patch("app.services.video_moderation_service._normalize_for_rekognition")
    @patch("app.services.video_moderation_service.detect_moderation_labels")
    @patch("app.services.video_moderation_service._extract_moderation_frames")
    @patch("app.services.video_moderation_service.get_ai_pipeline_config")
    def test_config_loaded_once_for_video_with_multiple_frames(
        self, mock_get_config, mock_extract_frames, mock_detect_labels, mock_normalize
    ):
        """
        Config đọc 1 lần cho VIDEO dù có nhiều frame (khoảng 10-30 frame).
        Mỗi frame có thể trả nhiều label vi phạm → BẪY: gọi DB per-frame × per-label

        Mock đúng hàm nội bộ _extract_moderation_frames() (video_moderation_service
        tự trích frame bằng ffmpeg, KHÔNG dùng extract_frames_from_video của module
        fingerprint) — patch nhầm tên hàm sẽ AttributeError ngay từ decorator.
        """
        # Arrange
        mock_normalize.return_value = b"normalized"
        mock_get_config.return_value = {
            "fingerprint_similarity_threshold": 0.90,
            "fingerprint_cluster_threshold": 0.95,
            "rekognition_confidence_threshold": 80.0,
            "rekognition_violence_confidence_threshold": 60.0,
            # _moderate_video đọc trực tiếp config["moderation_frame_interval"] khi tính
            # endTimestampMs cho violation — thiếu key này sẽ KeyError khi có vi phạm.
            "moderation_frame_interval": 2.0,
        }

        # Mock video extraction: 5 frame (timestamp_sec, frame_bytes) — đúng shape trả về
        # thật của _extract_moderation_frames().
        mock_extract_frames.return_value = [
            (0.0, b"frame_1_bytes"),
            (2.0, b"frame_2_bytes"),
            (4.0, b"frame_3_bytes"),
            (6.0, b"frame_4_bytes"),
            (8.0, b"frame_5_bytes"),
        ]

        # Mock Rekognition: mỗi frame trả 2-3 labels vi phạm — shape đã chuẩn hóa.
        mock_detect_labels.side_effect = [
            [{"name": "Adult", "confidence": 85.0, "parent_name": None}],
            [{"name": "Violence", "confidence": 90.0, "parent_name": None}],
            [{"name": "Adult", "confidence": 80.0, "parent_name": None}],
            [],  # Frame 4: an toàn
            [{"name": "Weapons", "confidence": 72.0, "parent_name": None}],
        ]

        # Act
        result = moderate_media(
            file_bytes=b"fake video data, 5 frames",
            media_type="VIDEO",
            media_id="test-video-1",
            correlation_id="corr-video",
        )

        # Assert
        # Config phải được gọi ĐÚNG 1 LẦN cho cả video (5 frames)
        assert mock_get_config.call_count == 1, (
            f"Video config được gọi {mock_get_config.call_count} lần, "
            "expected 1 lần (BẪY: nếu per-frame sẽ là 5 lần, nếu per-label sẽ lên 15+)"
        )
        assert result["success"] is True

    @patch("app.services.video_moderation_service._normalize_for_rekognition")
    @patch("app.services.video_moderation_service.detect_moderation_labels")
    @patch("app.services.video_moderation_service.get_ai_pipeline_config")
    def test_config_passed_to_threshold_function(
        self, mock_get_config, mock_detect_labels, mock_normalize
    ):
        """Config được truyền xuống _threshold_for_label(), không gọi DB trong hàm"""
        # Arrange
        mock_normalize.return_value = b"normalized"
        config_dict = {
            "fingerprint_similarity_threshold": 0.90,
            "fingerprint_cluster_threshold": 0.95,
            "rekognition_confidence_threshold": 85.0,  # Cao hơn default
            "rekognition_violence_confidence_threshold": 65.0,  # Cao hơn default
        }
        mock_get_config.return_value = config_dict

        mock_detect_labels.return_value = [
            {"name": "Violence", "confidence": 88.0, "parent_name": None},
        ]

        # Act
        result = moderate_media(
            file_bytes=b"violent content",
            media_type="IMAGE",
            media_id="test-violent",
            correlation_id="corr-violent",
        )

        # Assert
        # Config được gọi 1 lần
        assert mock_get_config.call_count == 1
        # Violence label dùng lower threshold (65.0): 88.0 > 65.0 → vi phạm
        assert result["success"] is True
        assert len(result["violations"]) > 0

    @patch("app.services.video_moderation_service._normalize_for_rekognition")
    @patch("app.services.video_moderation_service.detect_moderation_labels")
    @patch("app.services.video_moderation_service.get_ai_pipeline_config")
    def test_config_never_called_per_label_during_iteration(
        self, mock_get_config, mock_detect_labels, mock_normalize
    ):
        """
        Chứng minh config KHÔNG được gọi bên trong vòng lặp xử lý labels.
        Nếu code vi phạm (gọi DB trong _threshold_for_label hoặc loop xử lý),
        test này sẽ fail.
        """
        # Arrange: Track gọi get_ai_pipeline_config
        mock_normalize.return_value = b"normalized"
        call_order = []

        def track_get_config():
            call_order.append("get_config")
            return {
                "fingerprint_similarity_threshold": 0.90,
                "fingerprint_cluster_threshold": 0.95,
                "rekognition_confidence_threshold": 80.0,
                "rekognition_violence_confidence_threshold": 60.0,
            }

        mock_get_config.side_effect = track_get_config

        # Mock Rekognition: 3 labels — shape đã chuẩn hóa.
        mock_detect_labels.return_value = [
            {"name": "Adult", "confidence": 85.0, "parent_name": None},
            {"name": "Suggestive", "confidence": 80.0, "parent_name": "Adult"},
            {"name": "Violence", "confidence": 88.0, "parent_name": None},
        ]

        # Act
        result = moderate_media(
            file_bytes=b"multi-violation image",
            media_type="IMAGE",
            media_id="test-multi-1",
            correlation_id="corr-m1",
        )

        # Assert
        # get_config được gọi ĐÚNG 1 lần, KHÔNG interleave với label processing
        assert call_order.count("get_config") == 1, (
            f"get_config được gọi {len(call_order)} lần, "
            "expected 1 lần tại đầu moderate_media()"
        )
        assert result["success"] is True

    @patch("app.services.video_moderation_service._normalize_for_rekognition")
    @patch("app.services.video_moderation_service.detect_moderation_labels")
    @patch("app.services.video_moderation_service.get_ai_pipeline_config")
    def test_all_violations_use_same_config(
        self, mock_get_config, mock_detect_labels, mock_normalize
    ):
        """
        Tất cả violations được đánh giá theo cùng 1 config (không thay đổi giữa labels).
        Nếu code sai gọi DB per-label, config có thể thay đổi giữa labels → test fail.
        """
        # Arrange
        mock_normalize.return_value = b"normalized"
        initial_config = {
            "fingerprint_similarity_threshold": 0.90,
            "fingerprint_cluster_threshold": 0.95,
            "rekognition_confidence_threshold": 75.0,
            "rekognition_violence_confidence_threshold": 55.0,
        }
        mock_get_config.return_value = initial_config

        # Mock Rekognition: nhiều label — shape đã chuẩn hóa.
        mock_detect_labels.return_value = [
            {"name": "Violence", "confidence": 70.0, "parent_name": None},
            {"name": "Suggestive", "confidence": 70.0, "parent_name": None},
        ]

        # Act
        result = moderate_media(
            file_bytes=b"content",
            media_type="IMAGE",
            media_id="test-config-stability",
            correlation_id="corr-stable",
        )

        # Assert
        # Config được gọi 1 lần
        assert mock_get_config.call_count == 1
        # Cả 2 labels được đánh giá với cùng config (không có gọi lại get_config)
        assert result["success"] is True
