from core.tasks.delivery_normalization import (
    strip_heartbeat_token,
    strip_trailing_notify_false,
    normalize_heartbeat_reply,
)

# ── strip_heartbeat_token ──


def test_no_token():
    result, skip = strip_heartbeat_token("hello world")
    assert result == "hello world"
    assert skip is False


def test_empty_text():
    result, skip = strip_heartbeat_token("")
    assert result == ""
    assert skip is True


def test_whitespace_only():
    result, skip = strip_heartbeat_token("   ")
    assert result == ""
    assert skip is True


def test_heartbeat_ok_alone():
    result, skip = strip_heartbeat_token("HEARTBEAT_OK")
    assert result == ""
    assert skip is True


def test_heartbeat_ok_with_extra():
    result, skip = strip_heartbeat_token("一切正常 HEARTBEAT_OK")
    assert result == ""
    assert skip is True


def test_no_reply_alone():
    result, skip = strip_heartbeat_token("NO_REPLY")
    assert result == ""
    assert skip is True


# ── bug 回归：Markdown 包裹的 token ──

def test_heartbeat_ok_markdown_bold():
    result, skip = strip_heartbeat_token("**HEARTBEAT_OK**")
    assert result == ""
    assert skip is True


def test_heartbeat_ok_markdown_code():
    result, skip = strip_heartbeat_token("`HEARTBEAT_OK`")
    assert result == ""
    assert skip is True


def test_heartbeat_ok_markdown_italic():
    result, skip = strip_heartbeat_token("*HEARTBEAT_OK*")
    assert result == ""
    assert skip is True


def test_heartbeat_ok_wrapped_and_text():
    result, skip = strip_heartbeat_token("系统正常 **HEARTBEAT_OK** 请忽略")
    assert result == ""
    assert skip is True


def test_text_too_long():
    text = "一切正常 HEARTBEAT_OK " + "x" * 300
    result, skip = strip_heartbeat_token(text)
    assert skip is False
    assert len(result) > 0


def test_html_stripped():
    result, skip = strip_heartbeat_token("<b>加粗</b> HEARTBEAT_OK")
    assert result == ""
    assert skip is True


def test_case_insensitive():
    result, skip = strip_heartbeat_token("heartbeat_ok")
    assert result == ""
    assert skip is True


# ── strip_trailing_notify_false ──


def test_notify_false_trailing():
    result, found = strip_trailing_notify_false("hello\nnotify=false")
    assert result == "hello"
    assert found is True


def test_no_notify_false():
    result, found = strip_trailing_notify_false("hello world")
    assert result == "hello world"
    assert found is False


# ── normalize_heartbeat_reply ──


def test_normalize_full_pipeline():
    result, skip = normalize_heartbeat_reply("一切正常 **HEARTBEAT_OK**\nnotify=false")
    assert result == ""
    assert skip is True
