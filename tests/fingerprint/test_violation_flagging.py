"""
Test ownership-aware violation flagging — matcher.match_image_violation() /
matcher.match_segments() (xem app/fingerprint/matcher.py).

Đây là nơi chứa bug đã báo cáo (T3): chủ sở hữu gốc bị báo vi phạm ngược khi re-upload.
"""

from app.fingerprint.matcher import match_image_violation, match_segments


def test_t2_clone_gets_flagged(hit_factory):
    """T2: B clone ảnh của A — search trả về dòng của A (sạch, original_creator_id=A).
    Uploader là B → B phải bị flag vi phạm."""
    hits = [hit_factory(media_id="a_original", creator_id="creatorA", score=0.95,
                         original_creator_id="creatorA", is_violation=False)]

    violations = match_image_violation(hits, uploader_creator_id="creatorB")

    assert len(violations) == 1
    assert violations[0]["source_media_id"] == "a_original"


def test_t3_owner_reupload_not_flagged_reported_bug(hit_factory):
    """T3 — bug đã báo cáo: A re-upload lại chính nội dung của mình, trong Milvus vẫn còn
    dòng bản sao của B (creator_id=B, nhưng original_creator_id=A). A KHÔNG được bị flag."""
    hits = [hit_factory(media_id="b_clone", creator_id="creatorB", score=0.95,
                         original_creator_id="creatorA", is_violation=True)]

    violations = match_image_violation(hits, uploader_creator_id="creatorA")

    assert violations == []


def test_t4_repeated_owner_reupload_stays_clean(hit_factory):
    """T4: A re-upload nhiều lần liên tiếp — mỗi lần đều phải sạch (không vi phạm), kể cả
    khi Milvus có nhiều dòng bản sao của nhiều creator khác cùng trỏ về original_creator_id=A."""
    hits = [
        hit_factory(media_id="b_clone", creator_id="creatorB", score=0.95,
                    original_creator_id="creatorA", is_violation=True),
        hit_factory(media_id="c_clone", creator_id="creatorC", score=0.94,
                    original_creator_id="creatorA", is_violation=True),
    ]

    for _ in range(3):
        violations = match_image_violation(hits, uploader_creator_id="creatorA")
        assert violations == []


def test_t7_video_partial_span_copy_flagged(hit_factory):
    """T7: B copy 1 đoạn 10 giây trong video của A. match_segments() phải flag đúng đoạn đó
    cho B; A re-upload lại thì không bị flag gì (dùng lại cùng dữ liệu, đổi uploader)."""
    query_fingerprints = [{"timestamp": float(t), "vector": b"\x00"} for t in range(10, 20)]
    hits = [
        hit_factory(media_id="a_original", creator_id="creatorA", timestamp=float(t),
                    score=0.95, original_creator_id="creatorA", is_violation=False, query_index=i)
        for i, t in enumerate(range(10, 20))
    ]

    # B upload đoạn giống — phải bị flag toàn bộ span 10 giây.
    violations_b = match_segments(query_fingerprints, hits, uploader_creator_id="creatorB")
    assert len(violations_b) == 1
    assert violations_b[0]["source_media_id"] == "a_original"

    # A re-upload lại chính nó — không bị flag gì (chủ sở hữu của chính source đó).
    violations_a = match_segments(query_fingerprints, hits, uploader_creator_id="creatorA")
    assert violations_a == []


def test_t8_derivative_flags_only_foreign_owned_portion(hit_factory):
    """T8: uploader sở hữu cụm X (dòng original_creator_id == uploader) nhưng trong cùng file
    lại chứa đoạn copy từ cụm Y (original_creator_id của người khác) — chỉ đoạn Y bị flag."""
    query_fingerprints = [{"timestamp": float(t), "vector": b"\x00"} for t in range(0, 12)]

    # 6 giây đầu: uploader sở hữu (own content, media_id X) — không được flag.
    own_hits = [
        hit_factory(media_id="own_x", creator_id="creatorA", timestamp=float(t), score=0.95,
                    original_creator_id="creatorA", is_violation=False, query_index=i)
        for i, t in enumerate(range(0, 6))
    ]
    # 6 giây sau: copy từ creatorB (cụm Y, không phải của uploader) — PHẢI bị flag.
    foreign_hits = [
        hit_factory(media_id="foreign_y", creator_id="creatorB", timestamp=float(t), score=0.95,
                    original_creator_id="creatorB", is_violation=False, query_index=i)
        for i, t in zip(range(6, 12), range(6, 12))
    ]

    violations = match_segments(
        query_fingerprints, own_hits + foreign_hits, uploader_creator_id="creatorA"
    )

    assert len(violations) == 1
    assert violations[0]["source_media_id"] == "foreign_y"
