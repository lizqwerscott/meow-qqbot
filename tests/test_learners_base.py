import pytest

from core.learners.base import cosine_similarity, sanitize_for_learners

# ── cosine_similarity ──


def test_cosine_similarity_identical():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_partial():
    result = cosine_similarity([1, 2], [2, 4])
    assert result == pytest.approx(1.0)


def test_cosine_similarity_empty():
    assert cosine_similarity([], [1, 2]) == 0.0
    assert cosine_similarity([1, 2], []) == 0.0


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0, 0], [1, 2]) == 0.0


# ── 回归：维度不匹配 ──

def test_cosine_similarity_dimension_mismatch():
    with pytest.raises(ValueError, match="维度不匹配"):
        cosine_similarity([1, 2, 3], [1, 2])


# ── sanitize_for_learners ──


def test_sanitize_removes_reply_prefix():
    assert sanitize_for_learners("[正在回复 用户A]\n你好") == "你好"


def test_sanitize_removes_emoji_marker():
    assert sanitize_for_learners("今天[表情:大笑]很开心") == "今天很开心"


def test_sanitize_removes_emotion_marker():
    assert sanitize_for_learners("太棒了[情绪:高兴]") == "太棒了"


def test_sanitize_removes_at_mention():
    assert sanitize_for_learners("@小明 快来") == "快来"


def test_sanitize_removes_json_like():
    assert sanitize_for_learners('{"type":"card","content":"test"}') == ""


def test_sanitize_returns_empty_for_empty():
    assert sanitize_for_learners("") == ""


def test_sanitize_preserves_normal_text():
    assert sanitize_for_learners("正常聊天内容") == "正常聊天内容"
