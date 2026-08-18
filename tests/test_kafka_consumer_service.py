"""Unit tests for app/kafka/kafka_consumer_service.py — entrypoint thật nhận Kafka message
từ Spring Boot, dispatch tới fingerprint/moderation/delete/debezium handler.

Dự án CHƯA cài pytest-asyncio (kiểm tra requirements.txt xác nhận) — để không thêm
dependency mới ngoài phạm vi yêu cầu, mọi coroutine ở đây được chạy qua asyncio.run()
thuần bên trong hàm test đồng bộ (def test_x(), không phải async def test_x()), đây là
cách chuẩn không cần thư viện ngoài.

KHÔNG test consume_loop() — mock toàn bộ vòng đời AIOKafkaConsumer (start/getmany/commit/
stop) cho 1 vòng lặp vô hạn là bài test tích hợp thật sự, không phải unit test, giá trị
thu được không tương xứng độ phức tạp phải mock.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.kafka import kafka_consumer_service as kcs


def run(coro):
    """Helper chạy 1 coroutine trong test đồng bộ — thay thế pytest-asyncio."""
    return asyncio.run(coro)


def make_msg(topic: str, value: dict):
    msg = MagicMock()
    msg.topic = topic
    msg.value = value
    return msg


VALID_PIPELINE_JOB = {
    "mediaId": "media-abc-123",
    "s3Key": "source/videos/media-abc-123/original.mp4",
    "s3Bucket": "test-bucket",
    "mediaType": "IMAGE",
    "correlationId": "corr-1",
    "requestedAt": "2026-08-18T00:00:00",
    "creatorId": "creator-1",
}


class TestErrorResultHelpers:
    def test_copyright_error_result_shape(self):
        result = kcs._copyright_error_result("media-1", "corr-1", "boom")
        assert result["mediaId"] == "media-1"
        assert result["correlationId"] == "corr-1"
        assert result["success"] is False
        assert result["errorMessage"] == "boom"
        assert result["violations"] == []

    def test_moderation_error_result_shape(self):
        result = kcs._moderation_error_result("media-1", "corr-1", "boom")
        assert result["mediaId"] == "media-1"
        assert result["isSafe"] is False
        assert result["success"] is False
        assert result["errorMessage"] == "boom"

    def test_safe_dump_truncates_long_payload(self):
        huge = {"key": "x" * 10000}
        dumped = kcs._safe_dump(huge)
        assert len(dumped) <= 500

    def test_safe_dump_handles_non_serializable_values(self):
        # default=str trong json.dumps — không throw dù có object lạ.
        class Weird:
            def __str__(self):
                return "weird-obj"

        dumped = kcs._safe_dump({"x": Weird()})
        assert "weird-obj" in dumped


class TestBuildSslContext:
    def test_returns_none_when_no_ca_file_configured(self):
        with patch.object(kcs.settings, "KAFKA_SSL_CAFILE", ""):
            assert kcs._build_ssl_context() is None

    def test_builds_context_when_ca_file_configured(self):
        with patch.object(kcs.settings, "KAFKA_SSL_CAFILE", "ca.pem"), \
                patch.object(kcs.settings, "KAFKA_SSL_CERTFILE", "cert.pem"), \
                patch.object(kcs.settings, "KAFKA_SSL_KEYFILE", "key.pem"), \
                patch("ssl.create_default_context") as mock_create_ctx:
            mock_ctx = MagicMock()
            mock_create_ctx.return_value = mock_ctx

            result = kcs._build_ssl_context()

            assert result is mock_ctx
            mock_create_ctx.assert_called_once_with(cafile="ca.pem")
            mock_ctx.load_cert_chain.assert_called_once_with(certfile="cert.pem", keyfile="key.pem")


class TestSendHungJobErrorResult:
    def test_routes_pipeline_job_topic_to_copyright_result(self):
        msg = make_msg(kcs.TOPIC_PIPELINE_JOB, {"mediaId": "media-1", "correlationId": "corr-1"})
        with patch.object(kcs, "send_copyright_result", new=AsyncMock()) as mock_send:
            run(kcs._send_hung_job_error_result(msg, 300))
        mock_send.assert_called_once()
        assert mock_send.call_args.args[0] == "media-1"

    def test_routes_moderation_job_topic_to_moderation_result(self):
        msg = make_msg(kcs.TOPIC_MODERATION_JOB, {"mediaId": "media-1", "correlationId": "corr-1"})
        with patch.object(kcs, "send_moderation_result", new=AsyncMock()) as mock_send:
            run(kcs._send_hung_job_error_result(msg, 300))
        mock_send.assert_called_once()
        assert mock_send.call_args.args[0] == "media-1"

    def test_noop_when_no_media_id(self):
        msg = make_msg(kcs.TOPIC_PIPELINE_JOB, {"correlationId": "corr-1"})
        with patch.object(kcs, "send_copyright_result", new=AsyncMock()) as mock_send:
            run(kcs._send_hung_job_error_result(msg, 300))
        mock_send.assert_not_called()


class TestDispatchJob:
    def test_routes_pipeline_job_topic_to_handler(self):
        msg = make_msg(kcs.TOPIC_PIPELINE_JOB, {"mediaType": "IMAGE"})
        with patch.object(kcs, "_process_pipeline_job", new=AsyncMock()) as mock_handler:
            run(kcs._dispatch_job(msg))
        mock_handler.assert_called_once_with(msg.value)

    def test_routes_moderation_job_topic_to_handler(self):
        msg = make_msg(kcs.TOPIC_MODERATION_JOB, {"mediaType": "IMAGE"})
        with patch.object(kcs, "_process_moderation_job", new=AsyncMock()) as mock_handler:
            run(kcs._dispatch_job(msg))
        mock_handler.assert_called_once_with(msg.value)

    def test_routes_media_delete_topic_to_handler(self):
        msg = make_msg(kcs.TOPIC_MEDIA_DELETE, {"mediaId": "media-1"})
        with patch.object(kcs, "_process_media_delete", new=AsyncMock()) as mock_handler:
            run(kcs._dispatch_job(msg))
        mock_handler.assert_called_once_with(msg.value)

    def test_routes_recommendation_sync_topic_to_handler(self):
        msg = make_msg(kcs.TOPIC_RECOMMENDATION_SYNC, {"payload": {}})
        with patch.object(kcs, "_process_debezium_series", new=AsyncMock()) as mock_handler:
            run(kcs._dispatch_job(msg))
        mock_handler.assert_called_once_with(msg.value)

    def test_timeout_triggers_hung_job_error_result(self):
        msg = make_msg(kcs.TOPIC_PIPELINE_JOB, {"mediaType": "IMAGE"})

        async def hangs_forever(*args, **kwargs):
            await asyncio.sleep(999)

        with patch.object(kcs, "_process_pipeline_job", side_effect=hangs_forever), \
                patch.object(kcs, "_JOB_PROCESSING_TIMEOUT_SECONDS", 0.01), \
                patch.object(kcs, "_send_hung_job_error_result", new=AsyncMock()) as mock_hung:
            run(kcs._dispatch_job(msg))

        mock_hung.assert_called_once()

    def test_generic_exception_is_swallowed_not_raised(self):
        # _dispatch_job() KHÔNG được để lỗi văng ra ngoài — consume_loop() dùng
        # asyncio.gather() đợi cả batch, 1 job lỗi không được làm hỏng cả batch.
        msg = make_msg(kcs.TOPIC_PIPELINE_JOB, {"mediaType": "IMAGE"})
        with patch.object(kcs, "_process_pipeline_job", new=AsyncMock(side_effect=RuntimeError("boom"))):
            run(kcs._dispatch_job(msg))  # không raise là pass

    def test_video_job_uses_longer_timeout(self):
        # asyncio.Semaphore không hỗ trợ mock trực tiếp acquire/release dễ dàng — kiểm tra
        # gián tiếp qua timeout: mediaType=VIDEO phải rẽ đúng nhánh is_video=True (cùng
        # nhánh chọn _VIDEO_JOB_SEMAPHORE), verify qua timeout dài hơn được truyền cho
        # asyncio.wait_for().
        msg = make_msg(kcs.TOPIC_PIPELINE_JOB, {"mediaType": "VIDEO"})

        async def instant(*args, **kwargs):
            return None

        with patch.object(kcs, "_process_pipeline_job", side_effect=instant), \
                patch("asyncio.wait_for", wraps=asyncio.wait_for) as mock_wait_for:
            run(kcs._dispatch_job(msg))

        # Video job phải dùng _VIDEO_JOB_PROCESSING_TIMEOUT_SECONDS (1200), không phải
        # _JOB_PROCESSING_TIMEOUT_SECONDS (300).
        called_timeout = mock_wait_for.call_args.kwargs.get("timeout")
        assert called_timeout == kcs._VIDEO_JOB_PROCESSING_TIMEOUT_SECONDS


class TestProcessPipelineJob:
    def test_invalid_schema_sends_error_result_with_media_id(self):
        bad_data = {"mediaId": "media-1"}  # thiếu s3Key/s3Bucket/mediaType/correlationId/requestedAt
        with patch.object(kcs, "send_copyright_result", new=AsyncMock()) as mock_send:
            run(kcs._process_pipeline_job(bad_data))

        mock_send.assert_called_once()
        assert mock_send.call_args.args[0] == "media-1"
        assert mock_send.call_args.args[1]["success"] is False

    def test_invalid_schema_without_media_id_does_not_send_anything(self):
        # Không có mediaId để route kết quả lỗi về — job "biến mất", chỉ log, KHÔNG throw.
        bad_data = {"s3Key": "x"}
        with patch.object(kcs, "send_copyright_result", new=AsyncMock()) as mock_send:
            run(kcs._process_pipeline_job(bad_data))
        mock_send.assert_not_called()

    def test_happy_path_sends_success_result(self):
        fingerprint_response = MagicMock()
        fingerprint_response.content_id = "CID-000001"
        fingerprint_response.is_duplicate = False
        fingerprint_response.overall_similarity = 0.0
        fingerprint_response.fingerprint_count = 1
        fingerprint_response.violations = []

        moderation_result = {"isSafe": True, "success": True}

        with patch.object(kcs, "download_from_s3", return_value=b"filebytes"), \
                patch.object(kcs, "process_fingerprint", return_value=fingerprint_response), \
                patch.object(kcs, "moderate_media", return_value=moderation_result), \
                patch.object(kcs, "send_moderation_result", new=AsyncMock()), \
                patch.object(kcs, "embed_image_watermark", return_value=b"watermarked"), \
                patch.object(kcs, "upload_to_s3"), \
                patch.object(kcs, "generate_image_preview", return_value=b"preview"), \
                patch.object(kcs, "send_copyright_result", new=AsyncMock()) as mock_send_copyright:
            run(kcs._process_pipeline_job(VALID_PIPELINE_JOB))

        mock_send_copyright.assert_called_once()
        sent_result = mock_send_copyright.call_args.args[1]
        assert sent_result["success"] is True
        assert sent_result["mediaId"] == "media-abc-123"
        assert sent_result["contentId"] == "CID-000001"

    def test_unsafe_moderation_does_not_stop_pipeline(self):
        # isSafe=False KHÔNG được return sớm — watermark/preview vẫn phải chạy để Admin
        # duyệt tay có đủ file (xem comment thật trong code production).
        fingerprint_response = MagicMock()
        fingerprint_response.content_id = "CID-000002"
        fingerprint_response.is_duplicate = False
        fingerprint_response.overall_similarity = 0.0
        fingerprint_response.fingerprint_count = 1
        fingerprint_response.violations = []

        moderation_result = {"isSafe": False, "success": True}

        with patch.object(kcs, "download_from_s3", return_value=b"filebytes"), \
                patch.object(kcs, "process_fingerprint", return_value=fingerprint_response), \
                patch.object(kcs, "moderate_media", return_value=moderation_result), \
                patch.object(kcs, "send_moderation_result", new=AsyncMock()), \
                patch.object(kcs, "embed_image_watermark", return_value=b"watermarked") as mock_embed, \
                patch.object(kcs, "upload_to_s3"), \
                patch.object(kcs, "generate_image_preview", return_value=b"preview"), \
                patch.object(kcs, "send_copyright_result", new=AsyncMock()) as mock_send_copyright:
            run(kcs._process_pipeline_job(VALID_PIPELINE_JOB))

        # Watermark step vẫn chạy dù isSafe=False.
        mock_embed.assert_called_once()
        sent_result = mock_send_copyright.call_args.args[1]
        assert sent_result["success"] is True  # job vẫn "thành công" về mặt kỹ thuật

    def test_watermark_failure_does_not_fail_whole_job(self):
        fingerprint_response = MagicMock()
        fingerprint_response.content_id = "CID-000003"
        fingerprint_response.is_duplicate = False
        fingerprint_response.overall_similarity = 0.0
        fingerprint_response.fingerprint_count = 1
        fingerprint_response.violations = []

        with patch.object(kcs, "download_from_s3", return_value=b"filebytes"), \
                patch.object(kcs, "process_fingerprint", return_value=fingerprint_response), \
                patch.object(kcs, "moderate_media", return_value={"isSafe": True, "success": True}), \
                patch.object(kcs, "send_moderation_result", new=AsyncMock()), \
                patch.object(kcs, "embed_image_watermark", side_effect=RuntimeError("watermark boom")), \
                patch.object(kcs, "generate_image_preview", return_value=b"preview"), \
                patch.object(kcs, "upload_to_s3"), \
                patch.object(kcs, "send_copyright_result", new=AsyncMock()) as mock_send_copyright:
            run(kcs._process_pipeline_job(VALID_PIPELINE_JOB))

        sent_result = mock_send_copyright.call_args.args[1]
        assert sent_result["success"] is True
        assert sent_result["watermarkedS3Key"] is None

    def test_fingerprint_exception_sends_error_result(self):
        with patch.object(kcs, "download_from_s3", side_effect=RuntimeError("S3 down")), \
                patch.object(kcs, "send_copyright_result", new=AsyncMock()) as mock_send_copyright:
            run(kcs._process_pipeline_job(VALID_PIPELINE_JOB))

        sent_result = mock_send_copyright.call_args.args[1]
        assert sent_result["success"] is False
        assert "S3 down" in sent_result["errorMessage"]


class TestProcessModerationJob:
    def test_invalid_schema_sends_error_result(self):
        bad_data = {"mediaId": "media-1"}
        with patch.object(kcs, "send_moderation_result", new=AsyncMock()) as mock_send:
            run(kcs._process_moderation_job(bad_data))

        mock_send.assert_called_once()
        assert mock_send.call_args.args[0] == "media-1"
        assert mock_send.call_args.args[1]["success"] is False

    def test_happy_path_sends_result(self):
        with patch.object(kcs, "download_from_s3", return_value=b"filebytes"), \
                patch.object(kcs, "moderate_media", return_value={"isSafe": True, "success": True}), \
                patch.object(kcs, "send_moderation_result", new=AsyncMock()) as mock_send:
            run(kcs._process_moderation_job(VALID_PIPELINE_JOB))

        mock_send.assert_called_once_with("media-abc-123", {"isSafe": True, "success": True})

    def test_exception_sends_error_result(self):
        with patch.object(kcs, "download_from_s3", side_effect=RuntimeError("boom")), \
                patch.object(kcs, "send_moderation_result", new=AsyncMock()) as mock_send:
            run(kcs._process_moderation_job(VALID_PIPELINE_JOB))

        sent_result = mock_send.call_args.args[1]
        assert sent_result["success"] is False
        assert "boom" in sent_result["errorMessage"]


class TestProcessMediaDelete:
    def test_deletes_fingerprint_when_media_id_present(self):
        with patch.object(kcs, "delete_fingerprint") as mock_delete:
            run(kcs._process_media_delete({"mediaId": "media-1"}))
        mock_delete.assert_called_once_with("media-1")

    def test_noop_when_no_media_id(self):
        with patch.object(kcs, "delete_fingerprint") as mock_delete:
            run(kcs._process_media_delete({}))
        mock_delete.assert_not_called()

    def test_exception_is_logged_not_raised(self):
        with patch.object(kcs, "delete_fingerprint", side_effect=RuntimeError("milvus down")):
            run(kcs._process_media_delete({"mediaId": "media-1"}))  # không raise là pass


class TestProcessDebeziumSeries:
    def _patch_recommendation_deps(self):
        return patch.multiple(
            "app.services.recommendation_service",
            process_series_deletion=MagicMock(return_value=["neighbor-1"]),
            recalculate_series=MagicMock(return_value=["sim-1"]),
            process_series_upsert=MagicMock(return_value=["similar-1"]),
        )

    def test_create_event_is_ignored(self):
        data = {"payload": {"op": "c"}}
        with patch("app.kafka.kafka_producer_service.send_recommendation_result", new=AsyncMock()) as mock_send:
            run(kcs._process_debezium_series(data))
        mock_send.assert_not_called()

    def test_delete_event_processes_deletion_and_symmetric_update(self):
        data = {"payload": {"op": "d", "before": {"series_id": "series-1"}}}
        with self._patch_recommendation_deps(), \
                patch("app.kafka.kafka_producer_service.send_recommendation_result", new=AsyncMock()) as mock_send:
            run(kcs._process_debezium_series(data))

        # 1 lần cho chính series bị xóa (action=DELETE) + 1 lần cho neighbor (action=UPSERT).
        assert mock_send.call_count == 2
        first_call = mock_send.call_args_list[0]
        assert first_call.kwargs.get("action") == "DELETE" or first_call.args[-1] == "DELETE"

    def test_delete_event_without_before_is_noop(self):
        data = {"payload": {"op": "d"}}
        with patch("app.kafka.kafka_producer_service.send_recommendation_result", new=AsyncMock()) as mock_send:
            run(kcs._process_debezium_series(data))
        mock_send.assert_not_called()

    def test_update_with_deleted_status_processes_as_deletion(self):
        data = {"payload": {"op": "u", "after": {"series_id": "series-1", "status": "DELETED"}}}
        with self._patch_recommendation_deps(), \
                patch("app.kafka.kafka_producer_service.send_recommendation_result", new=AsyncMock()) as mock_send:
            run(kcs._process_debezium_series(data))

        assert mock_send.call_count == 2  # DELETE + symmetric neighbor UPSERT

    def test_update_with_is_deleted_flag_processes_as_deletion(self):
        data = {"payload": {"op": "u", "after": {"series_id": "series-1", "status": "PUBLISHED", "is_deleted": True}}}
        with self._patch_recommendation_deps(), \
                patch("app.kafka.kafka_producer_service.send_recommendation_result", new=AsyncMock()) as mock_send:
            run(kcs._process_debezium_series(data))

        assert mock_send.call_count == 2

    def test_update_with_published_status_processes_upsert(self):
        series_id = "series-upsert-test"
        data = {"payload": {"op": "u", "after": {"series_id": series_id, "status": "PUBLISHED"}}}
        with self._patch_recommendation_deps(), \
                patch("app.db.mongodb.get_series_metadata", new=AsyncMock(return_value={
                    "category": [], "tags": [], "ageRating": "", "language": "vi",
                })), \
                patch("app.kafka.kafka_producer_service.send_recommendation_result", new=AsyncMock()) as mock_send:
            run(kcs._process_debezium_series(data))

        assert mock_send.call_count >= 1
        first_call_args = mock_send.call_args_list[0]
        assert first_call_args.args[0] == series_id

    def test_update_with_unsupported_status_is_ignored(self):
        # "DRAFT"/"HIDDEN"/"FORCE_HIDDEN" đều nằm trong nhóm bị coi LÀ deletion (xem code
        # thật) — dùng 1 status hoàn toàn không thuộc cả 2 nhóm (deletion lẫn
        # PUBLISHED/SCHEDULED) để test đúng nhánh "ignored" thật sự.
        data = {"payload": {"op": "u", "after": {"series_id": "series-1", "status": "ARCHIVED"}}}
        with patch("app.kafka.kafka_producer_service.send_recommendation_result", new=AsyncMock()) as mock_send:
            run(kcs._process_debezium_series(data))
        mock_send.assert_not_called()

    def test_update_without_after_is_noop(self):
        data = {"payload": {"op": "u"}}
        with patch("app.kafka.kafka_producer_service.send_recommendation_result", new=AsyncMock()) as mock_send:
            run(kcs._process_debezium_series(data))
        mock_send.assert_not_called()

    def test_debounce_skips_duplicate_upsert_within_window(self):
        series_id = "series-debounce-test"
        data = {"payload": {"op": "u", "after": {"series_id": series_id, "status": "PUBLISHED"}}}

        # Giả lập series này vừa được xử lý 1 giây trước (trong cửa sổ debounce 30s).
        kcs._last_series_upsert_at[series_id] = kcs.time.monotonic()
        try:
            with self._patch_recommendation_deps(), \
                    patch("app.db.mongodb.get_series_metadata", new=AsyncMock()) as mock_get_meta, \
                    patch("app.kafka.kafka_producer_service.send_recommendation_result", new=AsyncMock()) as mock_send:
                run(kcs._process_debezium_series(data))

            mock_get_meta.assert_not_called()
            mock_send.assert_not_called()
        finally:
            kcs._last_series_upsert_at.pop(series_id, None)

    def test_exception_is_caught_not_raised(self):
        data = {"payload": {"op": "u", "after": {"series_id": "s1", "status": "PUBLISHED"}}}
        with patch("app.db.mongodb.get_series_metadata", new=AsyncMock(side_effect=RuntimeError("mongo down"))):
            run(kcs._process_debezium_series(data))  # không raise là pass
