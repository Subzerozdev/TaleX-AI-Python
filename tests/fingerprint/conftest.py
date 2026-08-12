"""
Fixtures dùng chung cho test bộ content-ownership registry (Content-ID pipeline).

Không cần Milvus thật cho các test unit — dùng dict giống HỆT shape trả về từ
milvus_store.search_similar() để test resolver/matcher độc lập, đúng tinh thần
"không mock che giấu logic thật" (xem plan phase-05).
"""

import pytest


def make_hit(
    media_id="m1",
    creator_id="creatorA",
    timestamp=0.0,
    score=0.99,
    content_cluster_id="cluster-1",
    original_creator_id="creatorA",
    first_seen_at=1000.0,
    is_violation=False,
    query_index=0,
):
    """Xây 1 dict giống hệt shape trả về từ milvus_store.search_similar()."""
    return {
        "query_index": query_index,
        "media_id": media_id,
        "creator_id": creator_id,
        "timestamp": timestamp,
        "score": score,
        "content_cluster_id": content_cluster_id,
        "original_creator_id": original_creator_id,
        "first_seen_at": first_seen_at,
        "is_violation": is_violation,
    }


@pytest.fixture
def hit_factory():
    """Factory tạo search-result dict — dùng trong test thay vì lặp lại dict thủ công."""
    return make_hit
