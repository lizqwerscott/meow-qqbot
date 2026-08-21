import io
import json
import struct
import wave
from email.parser import BytesParser
from email.policy import default

import httpx
import numpy as np
import pytest

from core.ai.tts_service import TtsService
from core.bootstrap import _resolve_tts_backend_and_base_url
from core.tools._types import ToolContext
from core.tools.deps import ToolDeps
from core.tools.impl.tts import create_tts_entries
from core.tools.ref import Ref


def _float_wav(samples: list[float], sample_rate: int = 44100) -> bytes:
    payload = np.asarray(samples, dtype="<f4").tobytes()
    fmt_chunk = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    chunks = b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
    chunks += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def _multipart_parts(request: httpx.Request):
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: "
        + request.headers["content-type"].encode("ascii")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + request.content
    )
    return {
        part.get_param("name", header="content-disposition"): part
        for part in message.iter_parts()
    }


def _multipart_fields(request: httpx.Request) -> dict[str, bytes]:
    return {
        name: part.get_payload(decode=True)
        for name, part in _multipart_parts(request).items()
    }


@pytest.mark.asyncio
async def test_s2_pro_uses_multipart_clone_request_and_returns_pcm_wav(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference audio")
    generated = _float_wav([0.25, -0.5])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == "http://s2.test/generate"
        assert request.headers["content-type"].startswith("multipart/form-data")
        fields = _multipart_fields(request)
        parts = _multipart_parts(request)
        assert set(parts) == {"text", "reference", "reference_text", "params"}
        assert fields["text"] == b"[excited] Hello"
        assert fields["reference"] == b"reference audio"
        assert parts["reference"].get_filename() == "reference.wav"
        assert parts["reference"].get_content_type() == "audio/x-wav"
        assert fields["reference_text"] == b"Reference transcript"
        assert json.loads(fields["params"]) == {"temperature": 0.58, "top_p": 0.88}
        return httpx.Response(200, content=generated, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = TtsService(
            base_url="http://s2.test",
            http_client=client,
            backend="s2-pro",
            ref_audio=str(reference),
            ref_text="Reference transcript",
            s2_params={"temperature": 0.58, "top_p": 0.88},
            temp_dir=str(tmp_path / "tts"),
        )
        result = await service.synthesize("[excited] Hello")

    assert result is not None
    with wave.open(io.BytesIO(result), "rb") as output:
        assert output.getframerate() == 44100
        assert output.getnchannels() == 1
        assert output.getsampwidth() == 2
        samples = np.frombuffer(output.readframes(2), dtype="<i2")
    np.testing.assert_allclose(samples, [15564, -31128], atol=1)


@pytest.mark.asyncio
async def test_s2_pro_without_reference_sends_only_text_and_params(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://s2.test/generate"
        fields = _multipart_fields(request)
        assert fields == {
            "text": b"Hello",
            "params": b'{"max_new_tokens": 1024}',
        }
        return httpx.Response(200, content=_float_wav([0.0]), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = TtsService(
            base_url="http://s2.test",
            http_client=client,
            backend="s2-pro",
            s2_params={"max_new_tokens": 1024},
            temp_dir=str(tmp_path / "tts"),
        )
        result = await service.synthesize("Hello")

    assert result is not None


@pytest.mark.asyncio
async def test_s2_pro_normalizes_restricted_text_and_blank_reference_transcript(
    tmp_path,
):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference audio")

    def handler(request: httpx.Request) -> httpx.Response:
        fields = _multipart_fields(request)
        assert fields == {"text": "[whispering，] Hello， quiet，calm".encode()}
        return httpx.Response(200, content=_float_wav([0.0]), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = TtsService(
            base_url="http://s2.test",
            http_client=client,
            backend="s2-pro",
            ref_audio=str(reference),
            ref_text=" \t ",
            temp_dir=str(tmp_path / "tts"),
        )
        result = await service.synthesize("[whispering；] Hello; (quiet)；（calm）")

    assert result is not None


@pytest.mark.asyncio
async def test_s2_pro_rejects_instructions_without_request(tmp_path):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=_float_wav([0.0]), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = TtsService(
            base_url="http://s2.test",
            http_client=client,
            backend="s2-pro",
            temp_dir=str(tmp_path / "tts"),
        )
        result = await service.synthesize("Hello", instructions="excited")

    assert result is None
    assert requests == []


@pytest.mark.asyncio
async def test_s2_pro_rejects_unsupported_voice_mode_without_request(tmp_path):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=_float_wav([0.0]), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = TtsService(
            base_url="http://s2.test",
            http_client=client,
            backend="s2-pro",
            temp_dir=str(tmp_path / "tts"),
        )
        result = await service.synthesize("Hello", voice_mode="creative")

    assert result is None
    assert requests == []


@pytest.mark.parametrize(
    ("backend", "default_base_url"),
    [
        ("s2-pro", "http://localhost:3030"),
        ("s2_pro", "http://localhost:3030"),
        ("S2-Pro", "http://localhost:3030"),
        ("voxcpm", "http://localhost:8080"),
    ],
)
def test_bootstrap_resolves_normalized_backend_defaults(backend, default_base_url):
    normalized_backend, resolved_base_url = _resolve_tts_backend_and_base_url(
        {"backend": backend}
    )

    assert resolved_base_url == default_base_url
    assert normalized_backend == backend.lower().replace("_", "-")


@pytest.mark.asyncio
async def test_voxcpm_request_preserves_preset_and_creative_modes(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference audio")
    expected_reference_audio = "cmVmZXJlbmNlIGF1ZGlv"
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == "http://voxcpm.test/v1/audio/speech"
        payloads.append(json.loads(request.content))
        return httpx.Response(200, content=b"voxcpm wav", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = TtsService(
            base_url="http://voxcpm.test",
            http_client=client,
            backend="voxcpm",
            model="voxcpm2",
            ref_audio=str(reference),
            cfg_value=1.5,
            inference_timesteps=12,
            temperature=0.7,
            seed=42,
            max_steps=99,
            temp_dir=str(tmp_path / "tts"),
        )
        preset_audio = await service.synthesize("Hello", instructions="cheerful")
        creative_audio = await service.synthesize(
            "Hello", instructions="calm", voice_mode="creative"
        )

    assert preset_audio == b"voxcpm wav"
    assert creative_audio == b"voxcpm wav"
    assert payloads[0] == {
        "model": "voxcpm2",
        "input": "(cheerful)Hello",
        "voice": "default",
        "response_format": "wav",
        "reference_audio": expected_reference_audio,
        "cfg_value": 1.5,
        "inference_timesteps": 12,
        "temperature": 0.7,
        "seed": 42,
        "max_steps": 99,
    }
    assert payloads[1] == {
        "model": "voxcpm2",
        "input": "(calm)Hello",
        "voice": "default",
        "response_format": "wav",
        "cfg_value": 1.5,
        "inference_timesteps": 12,
        "temperature": 0.7,
        "seed": 42,
        "max_steps": 99,
    }


@pytest.mark.asyncio
async def test_s2_pro_tool_preserves_text_tag_and_ignores_instructions(tmp_path):
    generated = _float_wav([0.0])
    requests = []

    class MediaUploader:
        async def upload(self, **kwargs):
            return {"file": "uploaded"}

    class BotEngine:
        def __init__(self):
            self.replies = []

        async def send_reply(self, **kwargs):
            self.replies.append(kwargs)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert (
            _multipart_fields(request)["text"]
            == "[soft tone] 主人主人，猫猫想要你摸摸头。好不好嘛？喵呜～".encode()
        )
        return httpx.Response(200, content=generated, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = TtsService(
            base_url="http://s2.test",
            http_client=client,
            backend="s2-pro",
            temp_dir=str(tmp_path / "tts"),
        )
        bot_engine = BotEngine()
        entry = create_tts_entries(
            ToolDeps(
                tts_service=Ref(service),
                media_uploader=Ref(MediaUploader()),
                bot_engine=Ref(bot_engine),
            )
        )[0]
        result = await entry.handler(
            {
                "instructions": "撒娇甜腻的语气，像小猫在主人怀里蹭蹭",
                "text": "[soft tone] 主人主人，猫猫想要你摸摸头。好不好嘛？喵呜～",
                "voice_mode": "preset",
            },
            ToolContext(
                chat_id="chat-id",
                is_group=False,
                reply_to="message-id",
                sender_id="user-id",
                reply_callback=lambda _content: None,
            ),
        )

    assert json.loads(result.content) == {
        "success": True,
        "message": "语音已发送到聊天中",
    }
    assert len(requests) == 1
    assert bot_engine.replies == [
        {
            "chat_id": "chat-id",
            "is_group": False,
            "message_id": "message-id",
            "media_file_info": {"file": "uploaded"},
        }
    ]
    async with httpx.AsyncClient() as client:
        s2_service = TtsService(
            base_url="http://s2.test",
            http_client=client,
            backend="s2-pro",
            temp_dir=str(tmp_path / "s2"),
        )
        s2_entry = create_tts_entries(ToolDeps(tts_service=Ref(s2_service)))[0]

        voxcpm_service = TtsService(
            base_url="http://voxcpm.test",
            http_client=client,
            backend="voxcpm",
            temp_dir=str(tmp_path / "voxcpm"),
        )
        voxcpm_entry = create_tts_entries(ToolDeps(tts_service=Ref(voxcpm_service)))[0]

    s2_text_rules = s2_entry.parameters["properties"]["text"]["description"]
    voxcpm_text_rules = voxcpm_entry.parameters["properties"]["text"]["description"]
    assert "S2-Pro" in s2_entry.description
    assert "[excited] [curious] [skeptical]" in s2_entry.description
    assert "[whispering] [shouting] [soft tone]" in s2_entry.description
    assert "[laughing] [chuckling] [sighing]" in s2_entry.description
    assert "每句最多一个标签" in s2_entry.description
    assert "长文本使用干净标点也可正常生成" in s2_text_rules
    assert "禁止分号和圆括号" in s2_text_rules
    assert "VoxCPM" not in s2_entry.description
    assert "instructions" not in s2_entry.parameters["properties"]
    assert s2_entry.parameters["properties"]["voice_mode"]["enum"] == ["preset"]
    assert "creative" not in s2_entry.description
    assert "VoxCPM" in voxcpm_entry.description
    assert "instructions" in voxcpm_entry.parameters["properties"]
    assert "[laughing]、[sigh]" in voxcpm_text_rules
    assert voxcpm_entry.parameters["properties"]["voice_mode"]["enum"] == [
        "preset",
        "creative",
    ]
    assert "S2-Pro" not in voxcpm_entry.description
