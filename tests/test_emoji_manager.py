import base64
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.managers.emoji_manager import EmojiManager


# 1×1 红色像素 PNG（base64）
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)


@pytest.fixture
def mock_http_client():
    client = MagicMock()
    resp = AsyncMock()
    resp.content = _TINY_PNG
    resp.raise_for_status = MagicMock()
    client.get = AsyncMock(return_value=resp)
    return client


@pytest.fixture
def mock_multimodal():
    mm = MagicMock()
    mm.analyze_emoji = AsyncMock(return_value=("测试表情", "这是一张测试图片", ["test"]))
    return mm


@pytest.fixture
async def emoji_mgr(mock_http_client, mock_multimodal, tmp_path):
    emoji_dir = str(tmp_path / "emojis")
    json_path = str(tmp_path / "emojis/emojis.json")
    mgr = EmojiManager(
        http_client=mock_http_client,
        multimodal_service=mock_multimodal,
        emoji_dir=emoji_dir,
        json_path=json_path,
    )
    return mgr


@pytest.mark.asyncio
async def test_get_or_build_non_gif(emoji_mgr, mock_multimodal):
    """non-GIF 图片不应产生 NameError (regression for bug #1)."""
    attachment = MagicMock()
    attachment.resolved_url = "https://example.com/test.png"
    attachment.content_type = "image/png"
    attachment.filename = "test.png"

    result = await emoji_mgr.get_or_build(attachment)

    assert result is not None
    assert len(result) == 4
    assert isinstance(result[0], str)
    mock_multimodal.analyze_emoji.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_or_build_non_gif_no_warning(emoji_mgr, mock_multimodal, caplog):
    """非 GIF 路径不应产生 VLM 相关 warning（regression for bug #1)."""
    caplog.set_level(logging.WARNING, logger="core.managers.emoji_manager")
    attachment = MagicMock()
    attachment.resolved_url = "https://example.com/test.png"
    attachment.content_type = "image/png"
    attachment.filename = "test.png"

    await emoji_mgr.get_or_build(attachment)

    vlm_warnings = [r for r in caplog.records if "VLM 分析表情失败" in r.getMessage()]
    assert len(vlm_warnings) == 0, f"不应有 VLM 警告: {vlm_warnings}"


@pytest.mark.asyncio
async def test_get_or_build_non_gif_no_multimodal(emoji_mgr):
    """没有 VLM 服务时也不应崩溃。"""
    emoji_mgr._multimodal = None
    attachment = MagicMock()
    attachment.resolved_url = "https://example.com/test.png"
    attachment.content_type = "image/png"
    attachment.filename = "test.png"

    result = await emoji_mgr.get_or_build(attachment)
    assert result is not None


@pytest.mark.asyncio
async def test_get_or_build_gif(emoji_mgr, mock_multimodal):
    """GIF 图片正常路径。"""
    attachment = MagicMock()
    attachment.resolved_url = "https://example.com/test.gif"
    attachment.content_type = "image/gif"
    attachment.filename = "test.gif"

    result = await emoji_mgr.get_or_build(attachment)
    assert result is not None


@pytest.mark.asyncio
async def test_get_or_build_cached(emoji_mgr, mock_multimodal):
    """已缓存的 emoji 不调用 VLM。"""
    # 先添加一个
    attachment = MagicMock()
    attachment.resolved_url = "https://example.com/test.png"
    attachment.content_type = "image/png"
    attachment.filename = "test.png"

    mock_multimodal.reset_mock()
    await emoji_mgr.get_or_build(attachment)
    assert mock_multimodal.analyze_emoji.await_count == 1

    # 第二次应该走缓存
    mock_multimodal.reset_mock()
    result = await emoji_mgr.get_or_build(attachment)
    assert mock_multimodal.analyze_emoji.await_count == 0
    assert len(result) == 4
