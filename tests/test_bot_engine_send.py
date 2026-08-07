from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core.engine.client import (
    BotEngine,
    _ReplyDeliveryState,
    _is_passive_reply_limit_error,
)


def make_engine(api):
    engine = object.__new__(BotEngine)
    engine.api = api
    engine._reply_delivery_states = {}
    return engine


def make_api():
    api = Mock()
    api.next_msg_seq.side_effect = iter(range(1, 10))
    api.build_text_body.side_effect = lambda content, **kwargs: SimpleNamespace(
        content=content, msg_id=kwargs.get("reply_to"), **kwargs
    )
    api.send_text = AsyncMock()
    api.post_group_message = AsyncMock(return_value={"id": "sent"})
    api.post_c2c_message = AsyncMock(return_value={"id": "sent"})
    return api


def test_passive_reply_limit_error_is_narrow():
    assert _is_passive_reply_limit_error(RuntimeError("被动回复时间或者次数超过限制"))
    assert not _is_passive_reply_limit_error(RuntimeError("QQ Bot API timeout"))
    assert not _is_passive_reply_limit_error(RuntimeError("bad request"))


@pytest.mark.asyncio
async def test_passive_limit_falls_back_once_without_reply_id():
    api = make_api()
    api.send_text.side_effect = RuntimeError("被动回复时间或者次数超过限制")
    engine = make_engine(api)

    await engine._send("group-1", "hello", reply_to="message-1", is_group=True)

    api.send_text.assert_awaited_once_with(
        "group", "group-1", "hello", reply_to="message-1", markdown=True, retries=1
    )
    api.post_group_message.assert_awaited_once()
    sent_msg = api.post_group_message.await_args.args[1]
    assert sent_msg.msg_id is None
    assert sent_msg.content == "hello"


@pytest.mark.asyncio
async def test_passive_limit_switches_all_following_chunks_to_proactive():
    api = make_api()
    api.send_text.side_effect = RuntimeError("被动回复时间或者次数超过限制")
    engine = make_engine(api)

    import core.engine.client as client_module

    original_split = client_module.split_markdown
    client_module.split_markdown = lambda content: ["one", "two"]
    try:
        await engine._send("group-1", "ignored", reply_to="message-1", is_group=True)
    finally:
        client_module.split_markdown = original_split

    assert api.send_text.await_count == 1
    assert api.post_group_message.await_count == 2
    sent_contents = [call.args[1].content for call in api.post_group_message.await_args_list]
    assert sent_contents == ["[1/2]\none", "[2/2]\ntwo"]
    assert all(call.args[1].msg_id is None for call in api.post_group_message.await_args_list)


@pytest.mark.asyncio
async def test_successful_passive_chunk_switches_following_chunks_to_proactive():
    api = make_api()
    engine = make_engine(api)

    import core.engine.client as client_module

    original_split = client_module.split_markdown
    client_module.split_markdown = lambda content: ["one", "two"]
    try:
        await engine._send("group-1", "ignored", reply_to="message-1", is_group=True)
    finally:
        client_module.split_markdown = original_split

    assert api.send_text.await_count == 1
    assert api.send_text.await_args.kwargs["reply_to"] == "message-1"
    assert api.post_group_message.await_count == 1
    assert api.post_group_message.await_args.args[1].msg_id is None


@pytest.mark.asyncio
async def test_non_passive_error_does_not_send_proactively():
    api = make_api()
    api.send_text.side_effect = RuntimeError("QQ Bot API timeout")
    engine = make_engine(api)

    with pytest.raises(RuntimeError):
        await engine._send("user-1", "hello", reply_to="message-1")

    api.post_c2c_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_markdown_error_retries_plain_text_once():
    api = make_api()
    api.send_text.side_effect = [
        RuntimeError("bad request"),
        {"id": "sent"},
    ]
    engine = make_engine(api)

    await engine._send("user-1", "hello", reply_to="message-1")

    assert api.send_text.await_count == 2
    assert api.send_text.await_args_list[0].kwargs["markdown"] is True
    assert api.send_text.await_args_list[1].kwargs["markdown"] is False
    api.post_c2c_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_passive_limit_falls_back_without_msg_id():
    api = make_api()
    api.post_group_message.side_effect = [
        RuntimeError("被动回复时间或者次数超过限制"),
        {"id": "sent"},
    ]
    engine = make_engine(api)

    await engine._send(
        "group-1",
        "",
        reply_to="message-1",
        is_group=True,
        media_file_info="file-token",
    )

    assert api.post_group_message.await_count == 2
    assert api.post_group_message.await_args_list[0].args[1].msg_id == "message-1"
    assert api.post_group_message.await_args_list[1].args[1].msg_id != "message-1"


@pytest.mark.asyncio
async def test_keyboard_passive_limit_falls_back_without_msg_id():
    api = make_api()
    api.post_c2c_message.side_effect = [
        RuntimeError("被动回复时间或者次数超过限制"),
        {"id": "sent"},
    ]
    engine = make_engine(api)

    await engine._send("user-1", "hello", reply_to="message-1", keyboard=Mock())

    assert api.post_c2c_message.await_count == 2
    assert api.post_c2c_message.await_args_list[0].args[1].msg_id == "message-1"
    assert api.post_c2c_message.await_args_list[1].args[1].msg_id is None


@pytest.mark.asyncio
async def test_stream_blocks_keep_proactive_mode_after_fallback():
    api = make_api()
    api.send_text.side_effect = RuntimeError("被动回复时间或者次数超过限制")
    engine = make_engine(api)

    await engine._send_reply("group-1", "first", "message-1", True)
    await engine._send_reply("group-1", "second", "message-1", True)

    assert api.send_text.await_count == 1
    assert api.post_group_message.await_count == 2
    assert all(
        call.args[1].msg_id is None for call in api.post_group_message.await_args_list
    )


@pytest.mark.asyncio
async def test_proactive_fallback_failure_does_not_retry():
    api = make_api()
    api.send_text.side_effect = RuntimeError("被动回复时间或者次数超过限制")
    api.post_group_message.side_effect = RuntimeError("proactive failed")
    engine = make_engine(api)

    with pytest.raises(RuntimeError, match="proactive failed"):
        await engine._send("group-1", "hello", reply_to="message-1", is_group=True)

    assert api.send_text.await_count == 1
    assert api.post_group_message.await_count == 1


@pytest.mark.asyncio
async def test_delivery_state_is_shared_by_send_reply_and_stream_callback():
    api = make_api()
    api.send_text.side_effect = RuntimeError("被动回复时间或者次数超过限制")
    engine = make_engine(api)

    await engine.send_reply("group-1", "first", message_id="message-1", is_group=True)
    await engine._send_reply("group-1", "second", "message-1", True)

    assert api.send_text.await_count == 1
    assert api.post_group_message.await_count == 2


def test_delivery_state_isolated_by_chat_context():
    engine = make_engine(make_api())

    group_state = engine._get_reply_delivery_state("group-1", "message-1", True)
    group_state.mode = "proactive"
    other_group_state = engine._get_reply_delivery_state("group-2", "message-1", True)
    c2c_state = engine._get_reply_delivery_state("group-1", "message-1", False)

    assert other_group_state.mode == "passive"
    assert c2c_state.mode == "passive"


def test_expired_delivery_states_are_removed(monkeypatch):
    engine = make_engine(make_api())
    state = _ReplyDeliveryState(last_used=100.0)
    engine._reply_delivery_states[("group", "old", "message")] = state
    monkeypatch.setattr("core.engine.client.time.monotonic", lambda: 800.0)

    engine._get_reply_delivery_state("group-1", "message-1", True)

    assert ("group", "old", "message") not in engine._reply_delivery_states
