"""
Test resolve_content_cluster() — bộ não của fix (xem app/fingerprint/content_ownership.py).

Không cần Milvus thật: monkeypatch search_similar() ở NGAY module content_ownership (nơi
tên đã được import vào), trả về dict giả lập nhưng đúng shape thật.
"""

import app.fingerprint.content_ownership as content_ownership
from app.fingerprint.content_ownership import resolve_content_cluster


def _patch_search(monkeypatch, hits: list[dict]):
    monkeypatch.setattr(content_ownership, "search_similar", lambda *a, **kw: hits)


def test_t1_new_upload_mints_new_cluster(monkeypatch):
    """T1: A uploads image (new) — không có match nào trong Milvus → tự tạo cụm mới, chủ sở
    hữu là chính creator đang upload."""
    _patch_search(monkeypatch, [])

    result = resolve_content_cluster([b"\x00" * 8], creator_id="creatorA", is_video=False)

    assert result.matched is False
    assert result.original_creator_id == "creatorA"
    assert result.cluster_id  # đã tự sinh uuid, không rỗng


def test_t5_owner_precedence_even_when_clone_rows_dominate(monkeypatch, hit_factory):
    """T5: dù top_k bị nhiều bản sao (is_violation=True) của các creator khác chiếm hết,
    resolver vẫn phải tìm đúng chủ sở hữu gốc (original_creator_id) ghi trên chính các dòng
    bản sao đó — không bị lạc hướng bởi số lượng bản sao áp đảo."""
    hits = [
        hit_factory(media_id="clone1", creator_id="creatorB", score=0.99,
                    original_creator_id="creatorA", is_violation=True, first_seen_at=2000.0),
        hit_factory(media_id="clone2", creator_id="creatorC", score=0.98,
                    original_creator_id="creatorA", is_violation=True, first_seen_at=3000.0),
        # Dòng "sạch" duy nhất — chủ sở hữu gốc thật.
        hit_factory(media_id="original", creator_id="creatorA", score=0.97,
                    original_creator_id="creatorA", is_violation=False, first_seen_at=1000.0),
    ]
    _patch_search(monkeypatch, hits)

    result = resolve_content_cluster([b"\x00" * 8], creator_id="creatorA", is_video=False)

    assert result.matched is True
    assert result.original_creator_id == "creatorA"
    assert result.first_seen_at == 1000.0


def test_t6_clone_of_clone_resolves_to_true_owner_not_intermediate_clone(monkeypatch, hit_factory):
    """T6: C copy lại bản sao của B (bản sao của A). Nếu cả dòng của A (sạch) lẫn dòng của B
    (bản sao) đều lọt vào kết quả search, resolver phải quy về A — không phải B — vì B
    is_violation=True bị loại khỏi ứng viên chủ sở hữu."""
    hits = [
        hit_factory(media_id="b_clone", creator_id="creatorB", score=0.99,
                    original_creator_id="creatorA", is_violation=True, first_seen_at=2000.0),
        hit_factory(media_id="a_original", creator_id="creatorA", score=0.98,
                    original_creator_id="creatorA", is_violation=False, first_seen_at=1000.0),
    ]
    _patch_search(monkeypatch, hits)

    result = resolve_content_cluster([b"\x00" * 8], creator_id="creatorC", is_video=False)

    assert result.matched is True
    assert result.original_creator_id == "creatorA"  # không phải creatorB


def test_t9_tie_break_deterministic_by_first_seen_at(monkeypatch, hit_factory):
    """T9: 2 ứng viên cùng vượt ngưỡng cluster nhưng thuộc 2 cụm khác nhau (race-like) —
    phải chọn quyết định (deterministic) theo first_seen_at sớm nhất, không ngẫu nhiên."""
    hits = [
        hit_factory(media_id="later", creator_id="creatorB", content_cluster_id="cluster-B",
                    score=0.97, original_creator_id="creatorB", first_seen_at=5000.0),
        hit_factory(media_id="earlier", creator_id="creatorA", content_cluster_id="cluster-A",
                    score=0.97, original_creator_id="creatorA", first_seen_at=1000.0),
    ]
    _patch_search(monkeypatch, hits)

    result = resolve_content_cluster([b"\x00" * 8], creator_id="creatorZ", is_video=False)

    assert result.cluster_id == "cluster-A"
    assert result.original_creator_id == "creatorA"


def test_t10_distinct_works_below_cluster_threshold_no_false_merge(monkeypatch, hit_factory):
    """T10: match có điểm dưới FINGERPRINT_CLUSTER_THRESHOLD (0.95) — dù trên ngưỡng vi phạm
    (0.90) — KHÔNG được coi là cùng cụm, tránh gộp nhầm 2 tác phẩm khác nhau vào 1 chủ sở hữu."""
    hits = [
        hit_factory(media_id="similar_but_different", creator_id="creatorB", score=0.92,
                    original_creator_id="creatorB", is_violation=False),
    ]
    _patch_search(monkeypatch, hits)

    result = resolve_content_cluster([b"\x00" * 8], creator_id="creatorA", is_video=False)

    assert result.matched is False  # tự tạo cụm mới, không gộp vào cụm của creatorB
    assert result.original_creator_id == "creatorA"


def test_video_resolves_to_cluster_with_most_frame_coverage(monkeypatch, hit_factory):
    """Nhánh video: cụm có nhiều query frame khớp nhất (coverage cao hơn) phải thắng, dù
    cụm kia có match với score cao hơn ở TỪNG frame riêng lẻ — coverage quyết định, không
    phải score đơn lẻ."""
    hits = [
        # cluster-A: khớp 8 frame liên tiếp (coverage=8).
        *[hit_factory(media_id="a1", creator_id="creatorA", content_cluster_id="cluster-A",
                       original_creator_id="creatorA", score=0.96, first_seen_at=1000.0,
                       query_index=i) for i in range(8)],
        # cluster-B: chỉ khớp 2 frame (coverage=2) dù score từng frame cao hơn.
        *[hit_factory(media_id="b1", creator_id="creatorB", content_cluster_id="cluster-B",
                       original_creator_id="creatorB", score=0.99, first_seen_at=500.0,
                       query_index=i) for i in range(2)],
    ]
    _patch_search(monkeypatch, hits)

    result = resolve_content_cluster([b"\x00" * 8] * 10, creator_id="creatorZ", is_video=True)

    assert result.matched is True
    assert result.cluster_id == "cluster-A"
    assert result.original_creator_id == "creatorA"


def test_video_coverage_tie_deterministic_by_earliest_first_seen_at(monkeypatch, hit_factory):
    """Đã sửa bug non-deterministic: 2 cụm HÒA coverage — phải luôn chọn cụm có first_seen_at
    sớm hơn, ổn định (deterministic) qua nhiều lần gọi, không phụ thuộc thứ tự duyệt dict."""
    hits = [
        *[hit_factory(media_id="later", creator_id="creatorB", content_cluster_id="cluster-later",
                       original_creator_id="creatorB", score=0.96, first_seen_at=5000.0,
                       query_index=i) for i in range(6)],
        *[hit_factory(media_id="earlier", creator_id="creatorA", content_cluster_id="cluster-earlier",
                       original_creator_id="creatorA", score=0.96, first_seen_at=1000.0,
                       query_index=i) for i in range(6)],
    ]
    _patch_search(monkeypatch, hits)

    # Chạy nhiều lần — phải luôn ra CÙNG 1 kết quả (deterministic).
    for _ in range(5):
        result = resolve_content_cluster([b"\x00" * 8] * 6, creator_id="creatorZ", is_video=True)
        assert result.cluster_id == "cluster-earlier"
        assert result.original_creator_id == "creatorA"


def test_video_below_min_frame_coverage_no_match(monkeypatch, hit_factory):
    """Coverage dưới ngưỡng FINGERPRINT_MIN_MATCH_SECONDS*FPS (mặc định 5 frame ở FPS=1) —
    không đủ để coi là cùng nội dung, phải tự tạo cụm mới."""
    hits = [
        hit_factory(media_id="a1", creator_id="creatorA", content_cluster_id="cluster-A",
                     original_creator_id="creatorA", score=0.96, query_index=0),
        hit_factory(media_id="a1", creator_id="creatorA", content_cluster_id="cluster-A",
                     original_creator_id="creatorA", score=0.96, query_index=1),
    ]
    _patch_search(monkeypatch, hits)

    result = resolve_content_cluster([b"\x00" * 8] * 2, creator_id="creatorZ", is_video=True)

    assert result.matched is False
