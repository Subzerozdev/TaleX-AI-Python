"""
Backfill Fingerprint Registry — migrate `talex_fingerprints_v3` → `_v4`.

Vấn đề: v4 thêm 4 field "sổ đăng ký chủ sở hữu nội dung gốc" (content_cluster_id,
original_creator_id, first_seen_at, is_violation — xem app/fingerprint/milvus_store.py) mà
~1912 dòng dữ liệu cũ trong v3 KHÔNG có. Script này BẮT BUỘC phải chạy trước khi coi migration
v3 → v4 là hoàn tất — không được bỏ dữ liệu cũ (quyết định của user, xem plan phase-01).

Thuật toán (tóm tắt, chi tiết xem plan phase-01 "Backfill Strategy"):
  1. Đọc TOÀN BỘ dòng v3 (query_iterator, không load 1 lần cả collection).
  2. Gom cụm greedy single-pass theo thứ tự `id` tăng dần (proxy cho thứ tự insert — Milvus
     không lưu insert timestamp thật), dùng ĐÚNG ngưỡng Hamming similarity
     FINGERPRINT_CLUSTER_THRESHOLD (0.95) như resolve_content_cluster() ở
     app/fingerprint/content_ownership.py — không phải luật riêng.
  3. Mỗi cụm: chủ sở hữu = dòng có `id` NHỎ NHẤT (proxy sớm nhất). Các dòng còn lại trong cụm
     → is_violation=True (bản sao theo mô hình mới).
  4. first_seen_at = thời điểm CHẠY SCRIPT (epoch giây) cho mọi dòng — không có timestamp
     lịch sử thật, dùng "now" đồng loạt để không bịa đặt độ chính xác giả.
  5. Insert lại TẤT CẢ dòng v3 (không bỏ, không gộp) vào v4 kèm 4 field derived ở trên.

QUAN TRỌNG — giới hạn đã biết (xem Risk Assessment trong plan phase-01): dùng `id` tự tăng của
Milvus làm proxy thứ tự upload là XẤP XỈ, không phải đảm bảo tuyệt đối. Chỉ ảnh hưởng ~1912
dòng legacy này (upload MỚI sau khi cutover sang v4 dùng first_seen_at thật, không qua script
này). Thuật toán gom cụm là single-pass greedy O(n*k), đủ dùng ở n≈1912 — nếu n lớn hơn nhiều
trước khi chạy lại script này, cần đánh giá lại (union-find pairwise đầy đủ), không nên chạy
mù thuật toán này ở quy mô rất lớn.

Cách chạy:
  python scripts/backfill_fingerprint_registry.py            # incremental (an toàn, bỏ qua
                                                               # media_id đã có trong v4)
  python scripts/backfill_fingerprint_registry.py --fresh    # xóa v4 hiện có, tạo lại từ đầu
                                                               # (dùng khi phát triển/test lại)
"""

import argparse
import sys
import time
import uuid

from loguru import logger
from pymilvus import Collection, connections, utility

sys.path.insert(0, ".")  # cho phép chạy trực tiếp `python scripts/...py` từ gốc repo

from app.core.config import settings  # noqa: E402
from app.fingerprint.hasher import VECTOR_DIM  # noqa: E402
from app.fingerprint import milvus_store  # noqa: E402

_V3_COLLECTION_NAME = "talex_fingerprints_v3"
_QUERY_BATCH_SIZE = 500


def _hamming_similarity(vec_a: bytes, vec_b: bytes) -> float:
    """Giống hệt công thức quy đổi trong milvus_store.search_similar() — giữ nhất quán."""
    a = int.from_bytes(vec_a, "big")
    b = int.from_bytes(vec_b, "big")
    hamming_distance = bin(a ^ b).count("1")
    return 1.0 - (hamming_distance / VECTOR_DIM)


def _load_all_v3_rows() -> list[dict]:
    """Đọc toàn bộ dòng v3 qua query_iterator (không load 1 lần cả collection vào RAM)."""
    if not utility.has_collection(_V3_COLLECTION_NAME):
        logger.warning(f"Collection '{_V3_COLLECTION_NAME}' không tồn tại — không có gì để backfill.")
        return []

    v3 = Collection(_V3_COLLECTION_NAME)
    v3.load()

    rows: list[dict] = []
    # media_id != "" luôn đúng với mọi dòng đã insert — dùng làm expr "lấy tất cả" an toàn
    # hơn giả định dấu của id tự tăng.
    iterator = v3.query_iterator(
        batch_size=_QUERY_BATCH_SIZE,
        expr='media_id != ""',
        output_fields=["id", "media_id", "creator_id", "timestamp", "vector"],
    )
    while True:
        batch = iterator.next()
        if not batch:
            iterator.close()
            break
        rows.extend(batch)

    rows.sort(key=lambda r: r["id"])  # thứ tự insert proxy — xem docstring module
    logger.info(f"Đã đọc {len(rows)} dòng từ '{_V3_COLLECTION_NAME}'.")
    return rows


def _cluster_rows(rows: list[dict]) -> list[list[dict]]:
    """
    Gom cụm greedy single-pass theo thứ tự `id` tăng dần — đại diện mỗi cụm là vector của
    dòng ĐẦU TIÊN được gán vào cụm đó (đơn giản, nhất quán với cách resolve_content_cluster()
    chọn 1 vector đại diện thay vì centroid trung bình).

    O(n * k) với k = số cụm đã tìm thấy — chấp nhận được ở n≈1912. Xem cảnh báo quy mô lớn
    ở docstring module.
    """
    clusters: list[dict] = []  # [{ "representative_vector": bytes, "members": [row, ...] }]

    for row in rows:
        best_cluster = None
        best_score = -1.0
        for cluster in clusters:
            score = _hamming_similarity(row["vector"], cluster["representative_vector"])
            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster is not None and best_score >= settings.FINGERPRINT_CLUSTER_THRESHOLD:
            best_cluster["members"].append(row)
        else:
            clusters.append({"representative_vector": row["vector"], "members": [row]})

    return [c["members"] for c in clusters]


def _existing_v4_media_to_cluster() -> dict[str, str]:
    """
    media_id -> content_cluster_id đã có sẵn trong v4 — dùng cho chế độ incremental.

    Trả cả cluster_id (không chỉ tập media_id) để nếu 1 cụm bị backfill DỞ DANG ở lần chạy
    trước (crash giữa chừng), lần chạy lại có thể TÁI SỬ DỤNG đúng content_cluster_id đã
    gán cho các thành viên đã insert, thay vì mint uuid mới cho phần còn lại — tránh 1 cụm
    nội dung bị TÁCH thành 2 content_cluster_id khác nhau giữa các thành viên của chính nó.
    """
    v4 = milvus_store.get_collection()
    rows = v4.query(expr='media_id != ""', output_fields=["media_id", "content_cluster_id"])
    return {r["media_id"]: r["content_cluster_id"] for r in rows}


def run_backfill(fresh: bool) -> None:
    if fresh and utility.has_collection(milvus_store._COLLECTION_NAME):
        logger.warning(f"--fresh: xóa collection '{milvus_store._COLLECTION_NAME}' hiện có trước khi tạo lại.")
        utility.drop_collection(milvus_store._COLLECTION_NAME)

    milvus_store.init_milvus()  # tạo v4 nếu chưa có (schema mới nhất, xem _create_collection)

    all_rows = _load_all_v3_rows()
    if not all_rows:
        logger.info("Không có dữ liệu v3 để backfill — kết thúc.")
        return

    existing_media_to_cluster: dict[str, str] = {} if fresh else _existing_v4_media_to_cluster()
    skip_media_ids = set(existing_media_to_cluster.keys())
    if skip_media_ids:
        logger.info(f"Chế độ incremental: bỏ qua {len(skip_media_ids)} media_id đã có trong v4.")

    clusters = _cluster_rows(all_rows)
    logger.info(f"Gom được {len(clusters)} cụm nội dung từ {len(all_rows)} dòng.")

    now = time.time()
    v4 = milvus_store.get_collection()

    media_ids, creator_ids, timestamps, vectors = [], [], [], []
    cluster_ids, original_creator_ids, first_seen_ats, is_violations = [], [], [], []
    per_cluster_summary = []
    inserted_count = 0
    skipped_count = 0

    for members in clusters:
        # Nếu BẤT KỲ thành viên nào của cụm này đã có trong v4 từ lần chạy trước (crash giữa
        # chừng) — tái sử dụng ĐÚNG content_cluster_id đã gán, không mint uuid mới. Nếu chưa
        # ai trong cụm được insert (lần chạy đầu, hoặc cụm hoàn toàn mới), mint như bình thường.
        reused_cluster_id = next(
            (existing_media_to_cluster[row["media_id"]] for row in members if row["media_id"] in existing_media_to_cluster),
            None,
        )
        content_cluster_id = reused_cluster_id or str(uuid.uuid4())
        owner_row = members[0]  # id nhỏ nhất trong cụm — xem docstring module
        original_creator_id = owner_row["creator_id"]

        member_media_ids = []
        for i, row in enumerate(members):
            member_media_ids.append(row["media_id"])
            if row["media_id"] in skip_media_ids:
                skipped_count += 1
                continue

            media_ids.append(row["media_id"])
            creator_ids.append(row["creator_id"])
            timestamps.append(row["timestamp"])
            vectors.append(row["vector"])
            cluster_ids.append(content_cluster_id)
            original_creator_ids.append(original_creator_id)
            first_seen_ats.append(now)
            is_violations.append(i != 0)  # dòng đầu tiên (chủ sở hữu) = False, còn lại = True
            inserted_count += 1

        per_cluster_summary.append(
            f"cluster={content_cluster_id} owner={original_creator_id} "
            f"members={len(members)} media_ids={member_media_ids}"
        )

    if media_ids:
        v4.insert(
            [
                media_ids,
                creator_ids,
                timestamps,
                vectors,
                cluster_ids,
                original_creator_ids,
                first_seen_ats,
                is_violations,
            ]
        )

    logger.info(f"Backfill xong: inserted={inserted_count}, skipped(đã có sẵn)={skipped_count}.")
    logger.info("=== Per-cluster summary (spot-check thủ công trước khi coi là final) ===")
    for line in per_cluster_summary:
        logger.info(line)

    v4_count = v4.num_entities
    v3_count = len(all_rows)
    if fresh and v4_count != v3_count:
        logger.warning(f"Số dòng KHÔNG khớp sau --fresh backfill: v4={v4_count}, v3={v3_count}.")
    else:
        logger.info(f"Số dòng v4 hiện tại: {v4_count} (v3 gốc có {v3_count} dòng).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Xóa collection v4 hiện có (nếu có) và backfill lại từ đầu. Mặc định: incremental "
        "(bỏ qua media_id đã có sẵn trong v4, an toàn để chạy lại nhiều lần).",
    )
    args = parser.parse_args()

    connections.connect(alias="default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
    run_backfill(fresh=args.fresh)
