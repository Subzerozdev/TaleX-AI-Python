"""
T12 — round-trip đầy đủ qua Milvus THẬT: A upload -> B clone (bị flag) -> A re-upload (sạch).

Bị skip tự động nếu không kết nối được Milvus (CI không có Milvus, hoặc chạy local không có
docker-compose lên) — unit suite ở các file khác mới là cổng chặn chính, đây chỉ xác nhận thêm
trên dữ liệu thật. Dùng media_id/creator_id ngẫu nhiên (uuid4) để không đụng dữ liệu prod thật,
và tự dọn dẹp sau khi test xong.
"""

import uuid

import pytest
from PIL import Image

from app.fingerprint.hasher import hash_image
from app.fingerprint.milvus_store import delete_by_media_id, init_milvus, is_connected
from app.services.fingerprint_service import process_fingerprint


def _fake_image_bytes(color: tuple[int, int, int]) -> bytes:
    import io

    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Gọi init_milvus() NGAY Ở MODULE LEVEL (không phải trong fixture) — pytestmark bên dưới cần
# is_connected() phản ánh đúng trạng thái TẠI THỜI ĐIỂM COLLECT test, còn fixture chỉ chạy
# sau khi collect xong, lúc đó is_connected() sẽ luôn False (chưa kịp gọi init_milvus()).
init_milvus()

pytestmark = pytest.mark.skipif(
    not is_connected(), reason="Milvus không khả dụng — bỏ qua integration test (xem unit suite)"
)


def test_t12_full_roundtrip_a_uploads_b_clones_a_reuploads():
    creator_a = f"test-creatorA-{uuid.uuid4().hex[:8]}"
    creator_b = f"test-creatorB-{uuid.uuid4().hex[:8]}"
    media_a1 = f"test-media-a1-{uuid.uuid4().hex[:8]}"
    media_b1 = f"test-media-b1-{uuid.uuid4().hex[:8]}"
    media_a2 = f"test-media-a2-{uuid.uuid4().hex[:8]}"

    # Màu solid ngẫu nhiên hóa nhẹ để không trùng fingerprint với dữ liệu test khác đang chạy
    # song song, nhưng vẫn pHash giống hệt giữa A và bản clone của B (cùng 1 ảnh y hệt).
    image_bytes = _fake_image_bytes((37, 201, 88))

    try:
        # A upload lần đầu — nội dung hoàn toàn mới, không vi phạm.
        resp_a1 = process_fingerprint(media_a1, creator_a, image_bytes, "a1.png")
        assert resp_a1.is_duplicate is False

        # B clone y hệt ảnh của A — phải bị flag vi phạm.
        resp_b1 = process_fingerprint(media_b1, creator_b, image_bytes, "b1.png")
        assert resp_b1.is_duplicate is True

        # A re-upload lại chính nội dung của mình (media_id khác, cùng nội dung) — KHÔNG
        # được bị flag dù bản sao của B vẫn còn trong Milvus. Đây là bug đã báo cáo.
        resp_a2 = process_fingerprint(media_a2, creator_a, image_bytes, "a2.png")
        assert resp_a2.is_duplicate is False, "Chủ sở hữu gốc bị báo vi phạm ngược — bug tái diễn"
    finally:
        delete_by_media_id(media_a1)
        delete_by_media_id(media_b1)
        delete_by_media_id(media_a2)
