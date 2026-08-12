"""
Test get_cluster_owner() — fallback tra chủ sở hữu cụm sau khi 1 media_id bị xóa (upsert),
đảm bảo quyền sở hữu KHÔNG mất chỉ vì 1 media_id cụ thể bị xóa (xem app/fingerprint/
content_ownership.py + plan phase-04 "Upsert & Delete-by-media_id Lifecycle Integrity").
"""

import app.fingerprint.content_ownership as content_ownership
from app.fingerprint.content_ownership import get_cluster_owner


def _patch_query(monkeypatch, rows: list[dict]):
    monkeypatch.setattr(content_ownership, "query_cluster_rows", lambda *a, **kw: rows)


def test_t11_owner_survives_via_other_rows_after_delete(monkeypatch):
    """T11: media_id của A bị xóa (upsert flow), nhưng các dòng KHÁC của cùng cluster (vd.
    A upload nhiều ảnh khác nhau đều thuộc cùng cluster, hoặc bản sao của B vẫn còn) vẫn còn
    trong Milvus — get_cluster_owner() phải tìm lại đúng original_creator_id=A, không mất."""
    # Dòng của B (bản sao, is_violation=True) KHÔNG được tính — chỉ dòng sạch mới hợp lệ,
    # nhưng query_cluster_rows(only_non_violation=True) đã tự lọc is_violation ở tầng Milvus,
    # nên rows giả lập ở đây chỉ chứa dòng sạch còn sót lại (dòng khác của chính A).
    rows = [{"original_creator_id": "creatorA", "first_seen_at": 1500.0}]
    _patch_query(monkeypatch, rows)

    result = get_cluster_owner("cluster-1")

    assert result is not None
    assert result.matched is True
    assert result.original_creator_id == "creatorA"
    assert result.first_seen_at == 1500.0


def test_t11_no_clean_rows_left_returns_none(monkeypatch):
    """Nếu cụm không còn dòng sạch nào (toàn bộ đã bị xóa hoặc chỉ còn bản sao vi phạm),
    get_cluster_owner() phải trả None — không tự bịa ra chủ sở hữu."""
    _patch_query(monkeypatch, [])

    result = get_cluster_owner("cluster-empty")

    assert result is None


def test_t4_first_seen_at_preserved_across_multiple_owner_rows(monkeypatch):
    """T4 (góc độ lifecycle): nếu cụm có nhiều dòng sạch với first_seen_at khác nhau (vd. A
    upload lại nhiều lần, mỗi lần resolver ghi lại đúng first_seen_at gốc), get_cluster_owner()
    phải luôn chọn first_seen_at SỚM NHẤT — không bị lần re-upload sau ghi đè."""
    rows = [
        {"original_creator_id": "creatorA", "first_seen_at": 3000.0},
        {"original_creator_id": "creatorA", "first_seen_at": 1000.0},  # sớm nhất — phải thắng
        {"original_creator_id": "creatorA", "first_seen_at": 2000.0},
    ]
    _patch_query(monkeypatch, rows)

    result = get_cluster_owner("cluster-1")

    assert result.first_seen_at == 1000.0
