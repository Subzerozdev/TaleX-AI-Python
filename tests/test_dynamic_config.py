"""
Test suite cho app/core/dynamic_config.py — đọc cấu hình AI pipeline động từ Postgres.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.core.dynamic_config import get_ai_pipeline_config, _defaults


class TestDynamicConfigDefaults:
    """Test hàm _defaults() trả về giá trị fallback từ config.py"""

    def test_defaults_returns_correct_shape(self):
        """_defaults() trả dict với đúng 13 key"""
        result = _defaults()
        assert isinstance(result, dict)
        assert len(result) == 13
        for key in (
            "fingerprint_similarity_threshold",
            "fingerprint_cluster_threshold",
            "rekognition_confidence_threshold",
            "rekognition_violence_confidence_threshold",
            "fingerprint_image_top_k",
            "fingerprint_video_top_k",
            "fingerprint_min_match_seconds",
            "fingerprint_max_gap_seconds",
            "fingerprint_fps",
            "fingerprint_max_frames",
            "fingerprint_max_file_size_mb",
            "rekognition_max_frames",
            "moderation_frame_interval",
        ):
            assert key in result

    def test_defaults_returns_numeric_values(self):
        """_defaults() trả float values"""
        result = _defaults()
        for key, value in result.items():
            assert isinstance(value, (int, float)), f"{key} không phải số: {value}"

    def test_defaults_values_reasonable(self):
        """_defaults() trả giá trị nằm trong khoảng hợp lý"""
        result = _defaults()
        # Similarity và cluster: [0, 1]
        assert 0 <= result["fingerprint_similarity_threshold"] <= 1
        assert 0 <= result["fingerprint_cluster_threshold"] <= 1
        # Rekognition: [0, 100]
        assert 0 <= result["rekognition_confidence_threshold"] <= 100
        assert 0 <= result["rekognition_violence_confidence_threshold"] <= 100


class TestGetAiPipelineConfigSuccess:
    """Test đọc thành công từ DB"""

    @patch("psycopg2.connect")
    def test_read_from_db_success(self, mock_connect):
        """Đọc 1 row từ DB → trả dict đúng shape"""
        # Arrange: Mock connection + cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Giả lập DB trả về 1 row với 13 giá trị (đúng order _COLUMNS)
        mock_cursor.fetchone.return_value = (
            0.88, 0.93, 78.0, 58.0, 25, 18, 6, 3, 2, 250, 80, 25, 1.5
        )

        # Act
        result = get_ai_pipeline_config()

        # Assert
        assert result["fingerprint_similarity_threshold"] == 0.88
        assert result["fingerprint_cluster_threshold"] == 0.93
        assert result["rekognition_confidence_threshold"] == 78.0
        assert result["rekognition_violence_confidence_threshold"] == 58.0
        assert result["fingerprint_image_top_k"] == 25
        assert result["fingerprint_video_top_k"] == 18
        assert result["fingerprint_min_match_seconds"] == 6
        assert result["fingerprint_max_gap_seconds"] == 3
        assert result["fingerprint_fps"] == 2
        assert result["fingerprint_max_frames"] == 250
        assert result["fingerprint_max_file_size_mb"] == 80
        assert result["rekognition_max_frames"] == 25
        assert result["moderation_frame_interval"] == 1.5

        # Verify connection được đóng
        mock_conn.close.assert_called_once()

    @patch("psycopg2.connect")
    def test_read_from_db_boundary_values(self, mock_connect):
        """Đọc giá trị biên từ DB"""
        # Arrange
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Giá trị biên cho 4 ngưỡng + 9 tham số kỹ thuật (>=1, interval nhỏ)
        mock_cursor.fetchone.return_value = (
            0.0, 1.0, 0.0, 100.0, 1, 1, 1, 1, 1, 1, 1, 1, 0.1
        )

        # Act
        result = get_ai_pipeline_config()

        # Assert
        assert result["fingerprint_similarity_threshold"] == 0.0
        assert result["fingerprint_cluster_threshold"] == 1.0
        assert result["rekognition_confidence_threshold"] == 0.0
        assert result["rekognition_violence_confidence_threshold"] == 100.0
        assert result["fingerprint_image_top_k"] == 1
        assert result["moderation_frame_interval"] == 0.1

    @patch("psycopg2.connect")
    def test_read_from_db_default_values_from_java(self, mock_connect):
        """Đọc default values giống Java AiPipelineConfig"""
        # Arrange: Java entity defaults
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Default từ Java entity (13 cột): 0.90, 0.95, 80.0, 60.0, 20, 15, 5, 2, 1, 300, 100, 30, 2.0
        mock_cursor.fetchone.return_value = (
            0.90, 0.95, 80.0, 60.0, 20, 15, 5, 2, 1, 300, 100, 30, 2.0
        )

        # Act
        result = get_ai_pipeline_config()

        # Assert
        assert result["fingerprint_similarity_threshold"] == 0.90
        assert result["fingerprint_cluster_threshold"] == 0.95
        assert result["rekognition_confidence_threshold"] == 80.0
        assert result["rekognition_violence_confidence_threshold"] == 60.0
        assert result["fingerprint_image_top_k"] == 20
        assert result["fingerprint_video_top_k"] == 15
        assert result["fingerprint_min_match_seconds"] == 5
        assert result["fingerprint_max_gap_seconds"] == 2
        assert result["fingerprint_fps"] == 1
        assert result["fingerprint_max_frames"] == 300
        assert result["fingerprint_max_file_size_mb"] == 100
        assert result["rekognition_max_frames"] == 30
        assert result["moderation_frame_interval"] == 2.0


class TestGetAiPipelineConfigFallback:
    """Test fallback về default khi DB lỗi hoặc bảng rỗng"""

    @patch("psycopg2.connect")
    def test_db_connection_refused_fallback(self, mock_connect):
        """Connection refused → fallback về default"""
        # Arrange: psycopg2 raise exception
        mock_connect.side_effect = Exception("connection refused")

        # Act
        result = get_ai_pipeline_config()

        # Assert: fallback về default
        defaults = _defaults()
        assert result == defaults

    @patch("psycopg2.connect")
    def test_db_timeout_fallback(self, mock_connect):
        """Connection timeout → fallback về default"""
        # Arrange
        mock_connect.side_effect = TimeoutError("connection timeout")

        # Act
        result = get_ai_pipeline_config()

        # Assert
        defaults = _defaults()
        assert result == defaults

    @patch("psycopg2.connect")
    def test_db_query_error_fallback(self, mock_connect):
        """Query execute error → fallback về default"""
        # Arrange
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # cursor.execute() raise exception
        mock_cursor.execute.side_effect = Exception("syntax error")

        # Act
        result = get_ai_pipeline_config()

        # Assert
        defaults = _defaults()
        assert result == defaults
        mock_conn.close.assert_called_once()

    @patch("psycopg2.connect")
    def test_db_table_empty_fallback(self, mock_connect):
        """Bảng rỗng (0 row) → fallback về default"""
        # Arrange
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # fetchone() trả None (không có row)
        mock_cursor.fetchone.return_value = None

        # Act
        result = get_ai_pipeline_config()

        # Assert
        defaults = _defaults()
        assert result == defaults
        mock_conn.close.assert_called_once()

    @patch("psycopg2.connect")
    def test_db_generic_error_fallback(self, mock_connect):
        """Generic error → fallback và KHÔNG raise exception"""
        # Arrange
        mock_connect.side_effect = RuntimeError("something went wrong")

        # Act
        result = get_ai_pipeline_config()

        # Assert: fallback về default, KHÔNG raise
        assert isinstance(result, dict)
        defaults = _defaults()
        assert result == defaults


class TestGetAiPipelineConfigReturnShape:
    """Test shape của return value"""

    @patch("psycopg2.connect")
    def test_return_shape_always_same(self, mock_connect):
        """Return shape luôn giống dù success hay fallback"""
        # Arrange: Success case
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (
            0.85, 0.90, 75.0, 55.0, 20, 15, 5, 2, 1, 300, 100, 30, 2.0
        )

        # Act: Success
        result_success = get_ai_pipeline_config()

        # Assert: shape OK (13 key)
        success_keys = set(result_success.keys())
        assert success_keys == {
            "fingerprint_similarity_threshold",
            "fingerprint_cluster_threshold",
            "rekognition_confidence_threshold",
            "rekognition_violence_confidence_threshold",
            "fingerprint_image_top_k",
            "fingerprint_video_top_k",
            "fingerprint_min_match_seconds",
            "fingerprint_max_gap_seconds",
            "fingerprint_fps",
            "fingerprint_max_frames",
            "fingerprint_max_file_size_mb",
            "rekognition_max_frames",
            "moderation_frame_interval",
        }

    @patch("psycopg2.connect")
    def test_return_values_typed_per_column(self, mock_connect):
        """Ngưỡng/interval ép về float; top_k/fps/frames ép về int (Milvus/phép chia cần int)."""
        # Arrange
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        # DB có thể trả int cho cột ngưỡng và ngược lại — verify ép kiểu đúng cả 2 chiều.
        mock_cursor.fetchone.return_value = (
            0.90, 0.95, 80, 60, 20.0, 15.0, 5.0, 2.0, 1.0, 300.0, 100.0, 30.0, 2
        )

        # Act
        result = get_ai_pipeline_config()

        # Assert
        float_keys = {
            "fingerprint_similarity_threshold",
            "fingerprint_cluster_threshold",
            "rekognition_confidence_threshold",
            "rekognition_violence_confidence_threshold",
            "moderation_frame_interval",
        }
        int_keys = {
            "fingerprint_image_top_k",
            "fingerprint_video_top_k",
            "fingerprint_min_match_seconds",
            "fingerprint_max_gap_seconds",
            "fingerprint_fps",
            "fingerprint_max_frames",
            "fingerprint_max_file_size_mb",
            "rekognition_max_frames",
        }
        for key in float_keys:
            assert isinstance(result[key], float), f"{key} phải là float: {result[key]}"
        for key in int_keys:
            assert isinstance(result[key], int), f"{key} phải là int: {result[key]}"


class TestGetAiPipelineConfigConnectionManagement:
    """Test quản lý connection"""

    @patch("psycopg2.connect")
    def test_connection_always_closed_on_success(self, mock_connect):
        """Connection được đóng sau khi đọc thành công"""
        # Arrange
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (
            0.90, 0.95, 80.0, 60.0, 20, 15, 5, 2, 1, 300, 100, 30, 2.0
        )

        # Act
        get_ai_pipeline_config()

        # Assert
        mock_conn.close.assert_called_once()

    @patch("psycopg2.connect")
    def test_connection_always_closed_on_error(self, mock_connect):
        """Connection được đóng ngay cả khi có error"""
        # Arrange
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.execute.side_effect = Exception("query failed")

        # Act
        get_ai_pipeline_config()

        # Assert: close vẫn được gọi
        mock_conn.close.assert_called_once()

    @patch("psycopg2.connect")
    def test_connection_closed_when_connection_fails(self, mock_connect):
        """Khi kết nối thất bại, không crash"""
        # Arrange
        mock_connect.side_effect = Exception("connection failed")

        # Act
        result = get_ai_pipeline_config()

        # Assert: không crash, trả default
        assert isinstance(result, dict)


class TestGetAiPipelineConfigIntegration:
    """Integration tests giả lập Postgres thực tế"""

    @patch("psycopg2.connect")
    def test_multiple_calls_independent(self, mock_connect):
        """Mỗi lần gọi mở connection độc lập (KHÔNG cache)"""
        # Arrange
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (
            0.90, 0.95, 80.0, 60.0, 20, 15, 5, 2, 1, 300, 100, 30, 2.0
        )

        # Act: gọi 3 lần
        get_ai_pipeline_config()
        get_ai_pipeline_config()
        get_ai_pipeline_config()

        # Assert: phải mở 3 connections (KHÔNG cache)
        assert mock_connect.call_count == 3
        assert mock_conn.close.call_count == 3

    @patch("psycopg2.connect")
    def test_sql_query_correct_columns(self, mock_connect):
        """SQL query đúng order cột theo AiPipelineConfig entity"""
        # Arrange
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (
            0.90, 0.95, 80.0, 60.0, 20, 15, 5, 2, 1, 300, 100, 30, 2.0
        )

        # Act
        get_ai_pipeline_config()

        # Assert: kiểm tra SQL được execute
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0][0]

        # Verify cột theo đúng order
        assert "fingerprint_similarity_threshold" in call_args
        assert "fingerprint_cluster_threshold" in call_args
        assert "rekognition_confidence_threshold" in call_args
        assert "rekognition_violence_confidence_threshold" in call_args
        assert "ai_pipeline_configs" in call_args
