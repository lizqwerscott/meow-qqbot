from types import SimpleNamespace

import pytest

from core.ai.deepseek_service import DeepSeekResponsesService
from core.ai.service import AIService


class FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None)
                )
            ],
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_openai_compatible_reasoning_effort_is_per_turn():
    service = AIService(
        api_key="test",
        model="chat-model",
        reasoning_effort="high",
    )
    client = FakeClient()
    service.client = client

    await service.chat_completion_with_tools(
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="low",
    )
    request = client.chat.completions.calls[-1]
    assert request["reasoning_effort"] == "low"
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert service.reasoning_effort == "high"

    await service.chat_completion_with_tools(
        messages=[{"role": "user", "content": "hello"}],
        model="deepseek-chat",
        reasoning_effort="none",
    )
    request = client.chat.completions.calls[-1]
    assert "reasoning_effort" not in request
    assert "extra_body" not in request
    assert request["temperature"] == 0.7


def test_deepseek_reasoning_effort_changes_only_request_projection():
    service = DeepSeekResponsesService(api_key="test", reasoning_effort="high")
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer"},
    ]

    inherited = service._build_kwargs(messages, None, service.model, None, None)
    overridden = service._build_kwargs(
        messages, None, service.model, None, None, reasoning_effort="low"
    )
    disabled = service._build_kwargs(
        messages, None, service.model, None, None, reasoning_effort="none"
    )

    assert inherited["reasoning"] == {"effort": "high"}
    assert overridden["reasoning"] == {"effort": "low"}
    assert "reasoning" not in disabled
    assert all(item["type"] != "reasoning" for item in disabled["input"])
