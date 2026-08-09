import asyncio
import sqlite3
from unittest.mock import AsyncMock

import pytest
from pypdf import PdfWriter

from core.media.service import MediaService
from core.media.store import MediaStore
from core.message import InputMessage, ResourceMeta


class FakeTranscriber:
    model_name = "small"

    def __init__(self):
        self.preload = AsyncMock()
        self.transcribe = AsyncMock(return_value="你好 hello")


class FakeOcrEngine:
    def __init__(self, lines):
        self._lines = lines
        self.calls = []

    async def recognize(self, image_path):
        self.calls.append(image_path)
        return list(self._lines)


@pytest.mark.asyncio
async def test_user_silk_voice_is_saved_as_wav_with_source_sample(
    tmp_path, monkeypatch
):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    await service.open()
    converted_path = tmp_path / "converted.wav"
    converted_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEconverted")
    converter = AsyncMock(return_value=str(converted_path))
    monkeypatch.setattr("core.media.service.convert_audio_to_wav", converter)
    resource = ResourceMeta(
        resource_type="voice",
        source_url="https://example.test/voice.amr",
        mime_type="audio/mp3",
        filename="voice.amr",
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="",
        is_group=True,
        resources=[resource],
    )
    service._download = AsyncMock(
        return_value=(b"\x02#!SILK_V3\x11\x00raw", "audio/mp3")
    )

    await service.ingest_message(message)

    converter.assert_awaited_once()
    assert resource.mime_type == "audio/wav"
    assert resource.storage_status == "ready"
    record = await service.store.authorize("g1", resource.media_uri)
    assert record is not None
    assert record.local_path.suffix == ".wav"
    assert record.local_path.read_bytes().startswith(b"RIFF")
    assert (
        record.local_path.with_name(f"{record.local_path.stem}.source.silk")
        .read_bytes()
        .startswith(b"\x02#!SILK_V3")
    )


@pytest.mark.asyncio
async def test_voice_conversion_error_falls_back_to_raw_silk(tmp_path, monkeypatch):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    await service.open()
    converter = AsyncMock(side_effect=RuntimeError("decoder unavailable"))
    monkeypatch.setattr("core.media.service.convert_audio_to_wav", converter)
    resource = ResourceMeta(
        resource_type="voice",
        source_url="https://example.test/voice.amr",
        mime_type="audio/mp3",
        filename="voice.amr",
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="",
        is_group=True,
        resources=[resource],
    )
    raw_data = b"\x02#!SILK_V3\x11\x00raw"
    service._download = AsyncMock(return_value=(raw_data, "audio/mp3"))

    await service.ingest_message(message)

    assert resource.storage_status == "ready"
    assert resource.mime_type == "audio/silk"
    record = await service.store.authorize("g1", resource.media_uri)
    assert record is not None
    assert record.local_path.suffix == ".silk"
    assert record.local_path.read_bytes() == raw_data


@pytest.mark.asyncio
async def test_file_silk_is_saved_without_conversion(tmp_path, monkeypatch):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    await service.open()
    converter = AsyncMock()
    monkeypatch.setattr("core.media.service.convert_audio_to_wav", converter)
    resource = ResourceMeta(
        resource_type="file",
        source_url="https://example.test/voice.amr",
        mime_type="audio/mp3",
        filename="voice.amr",
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="",
        is_group=True,
        resources=[resource],
    )
    raw_data = b"\x02#!SILK_V3\x11\x00raw"
    service._download = AsyncMock(return_value=(raw_data, "audio/mp3"))

    await service.ingest_message(message)

    converter.assert_not_awaited()
    assert resource.mime_type == "audio/mp3"
    record = await service.store.authorize("g1", resource.media_uri)
    assert record is not None
    assert record.local_path.suffix == ".amr"
    assert record.local_path.read_bytes() == raw_data


@pytest.mark.asyncio
async def test_image_understanding_disabled_hides_tools(tmp_path):
    service = MediaService(
        http_client=AsyncMock(),
        multimodal=AsyncMock(),
        storage_dir=tmp_path,
        image_understanding={"enabled": False},
    )
    assert not service.image_tools_enabled
    assert service.file_tools_enabled


@pytest.mark.asyncio
async def test_inspect_file_authorizes_type_and_truncation(tmp_path):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/notes.md",
        mime_type="text/markdown",
        filename="notes.md",
        data=b"abcdef",
    )
    result = await service.inspect_file(
        chat_id="g1", media_uri=record.media_uri, max_chars=4
    )
    assert result.content == "abcd"
    assert result.truncated
    assert (
        await service.read_file(chat_id="g2", media_uri=record.media_uri)
    ).error == ("MEDIA_FORBIDDEN")


@pytest.mark.asyncio
async def test_image_and_voice_report_forbidden_media(tmp_path):
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        multimodal=AsyncMock(),
    )
    await service.open()
    image = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="image",
        source_url="https://example.test/image.png",
        mime_type="image/png",
        filename="image.png",
        data=b"image",
    )
    assert (
        await service.inspect_image(
            chat_id="g2", media_uri=image.media_uri, question="描述"
        )
    ).error == "MEDIA_FORBIDDEN"


@pytest.mark.asyncio
async def test_read_file_is_the_public_attachment_reader(tmp_path):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/notes.md",
        mime_type="text/markdown",
        filename="notes.md",
        data=b"abcdef",
    )
    result = await service.read_file(chat_id="g1", media_uri=record.media_uri)
    assert result.content == "abcdef"


@pytest.mark.asyncio
async def test_inspect_file_rejects_unsupported_type(tmp_path):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/archive.zip",
        mime_type="application/zip",
        filename="archive.zip",
        data=b"not-a-zip",
    )
    result = await service.inspect_file(chat_id="g1", media_uri=record.media_uri)
    assert result.error == "UNSUPPORTED_MEDIA_TYPE"


@pytest.mark.asyncio
async def test_prepare_for_ai_includes_current_text_file_excerpt(tmp_path):
    service = MediaService(
        http_client=AsyncMock(), storage_dir=tmp_path, file_context_max_chars=4
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/notes.txt",
        mime_type="text/plain",
        filename="notes.txt",
        data=b"abcdef",
    )
    resource = ResourceMeta(
        resource_type="file",
        media_uri=record.media_uri,
        filename="notes.txt",
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="看看文件",
        is_group=True,
        resources=[resource],
    )
    context = await service.prepare_for_ai(message)
    text = context.as_text()
    assert "[文件摘要]" in text
    assert "文件未自动摘要" in text
    assert "read_file" in text


@pytest.mark.asyncio
async def test_prepare_for_ai_uses_cached_file_summary(tmp_path):
    ai_service = type("AI", (), {})()
    ai_service.model = "summary-model"
    ai_service.chat_completion = AsyncMock(return_value=("这是一份摘要", None))
    service = MediaService(
        http_client=AsyncMock(), storage_dir=tmp_path, ai_service=ai_service
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/notes.txt",
        mime_type="text/plain",
        filename="notes.txt",
        data=b"abcdef",
    )
    resource = ResourceMeta(
        resource_type="file", media_uri=record.media_uri, filename="notes.txt"
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="看看文件",
        is_group=True,
        resources=[resource],
    )

    first = await service.prepare_for_ai(message)
    second = await service.prepare_for_ai(message)
    stored = await service.store.authorize("g1", record.media_uri)

    assert "这是一份摘要" in first.as_text()
    assert second.as_text() == first.as_text()
    assert stored.file_summary == "这是一份摘要"
    assert ai_service.chat_completion.await_count == 1


@pytest.mark.asyncio
async def test_file_summary_failure_keeps_read_file_available(tmp_path):
    ai_service = type("AI", (), {})()
    ai_service.model = "summary-model"
    ai_service.chat_completion = AsyncMock(side_effect=RuntimeError("offline"))
    service = MediaService(
        http_client=AsyncMock(), storage_dir=tmp_path, ai_service=ai_service
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/notes.txt",
        mime_type="text/plain",
        filename="notes.txt",
        data=b"abcdef",
    )
    resource = ResourceMeta(resource_type="file", media_uri=record.media_uri)
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="摘要",
        is_group=True,
        resources=[resource],
    )

    context = await service.prepare_for_ai(message)
    original = await service.read_file(chat_id="g1", media_uri=record.media_uri)

    assert "可调用 read_file" in context.as_text()
    assert original.content == "abcdef"


@pytest.mark.asyncio
async def test_inspect_file_reuses_cached_text_extraction(tmp_path):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/notes.txt",
        mime_type="text/plain",
        filename="notes.txt",
        data=b"abcdef",
    )
    provider = service.file_capability.providers[0]
    execute = AsyncMock(return_value="abcdef")
    provider.execute = execute

    first = await service.inspect_file(chat_id="g1", media_uri=record.media_uri)
    second = await service.inspect_file(chat_id="g1", media_uri=record.media_uri)

    assert first.content == second.content == "abcdef"
    execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_file_inspection_uses_single_flight(tmp_path):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/notes.txt",
        mime_type="text/plain",
        filename="notes.txt",
        data=b"abcdef",
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def extract(*args, **kwargs):
        started.set()
        await release.wait()
        return "abcdef"

    provider = service.file_capability.providers[0]
    provider.execute = AsyncMock(side_effect=extract)
    first = asyncio.create_task(
        service.inspect_file(chat_id="g1", media_uri=record.media_uri)
    )
    await started.wait()
    second = asyncio.create_task(
        service.inspect_file(chat_id="g1", media_uri=record.media_uri)
    )
    release.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.content == second_result.content == "abcdef"
    provider.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_inspect_pdf_authorizes_pdf_and_returns_analysis(tmp_path):
    ai_service = type("AI", (), {})()
    ai_service.model = "summary-model"
    ai_service.chat_completion = AsyncMock(return_value=("分析结果", None))
    service = MediaService(
        http_client=AsyncMock(), storage_dir=tmp_path, ai_service=ai_service
    )
    await service.open()
    pdf_path = tmp_path / "report.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/report.pdf",
        mime_type="application/pdf",
        filename="report.pdf",
        data=pdf_path.read_bytes(),
    )
    provider = service.pdf_capability.providers[0]
    provider.execute = AsyncMock(return_value="页数: 2\n分析结果")

    result = await service.inspect_pdf(
        chat_id="g1", media_uri=record.media_uri, prompt="总结"
    )

    assert result.analysis == "分析结果"
    assert result.pages == 2
    provider.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_inspect_pdf_rejects_pdf_filename_with_wrong_mime(tmp_path):
    ai_service = type("AI", (), {})()
    ai_service.model = "summary-model"
    service = MediaService(
        http_client=AsyncMock(), storage_dir=tmp_path, ai_service=ai_service
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/report.pdf",
        mime_type="text/plain",
        filename="report.pdf",
        data=b"not a pdf",
    )

    result = await service.inspect_pdf(
        chat_id="g1", media_uri=record.media_uri, prompt="总结"
    )

    assert result.error == "UNSUPPORTED_MEDIA_TYPE"


@pytest.mark.asyncio
async def test_inspect_pdf_enforces_page_limit(tmp_path):
    ai_service = type("AI", (), {})()
    ai_service.model = "summary-model"
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        ai_service=ai_service,
        pdf_max_pages=1,
    )
    await service.open()
    pdf_path = tmp_path / "report.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/report.pdf",
        mime_type="application/pdf",
        filename="report.pdf",
        data=pdf_path.read_bytes(),
    )

    result = await service.inspect_pdf(
        chat_id="g1", media_uri=record.media_uri, prompt="总结"
    )

    assert result.error == "PDF_TOO_MANY_PAGES"


@pytest.mark.asyncio
async def test_current_prompt_version_replaces_persisted_stale_summary(tmp_path):
    multimodal = type("Multimodal", (), {})()
    multimodal.model = "vision-model"
    multimodal.analyze_image = AsyncMock(return_value="新摘要")
    service = MediaService(
        http_client=AsyncMock(), multimodal=multimodal, storage_dir=tmp_path
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="image",
        source_url="https://example.test/image.png",
        mime_type="image/png",
        filename="image.png",
        data=b"image",
    )
    await service.store.update_summary(record.media_id, "旧摘要", "old", "v1")
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="看看图片",
        is_group=True,
        resources=[ResourceMeta(resource_type="image", media_uri=record.media_uri)],
    )

    context = await service.prepare_for_ai(message)
    refreshed = await service.store.authorize("g1", record.media_uri, image_only=True)

    assert "新摘要" in context.as_text()
    assert refreshed.summary_version == "v2"
    multimodal.analyze_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_summary_uses_ocr_text_when_multimodal_absent(tmp_path):
    ocr = FakeOcrEngine(["这是截图里的第一行", "这是第二行内容"])
    service = MediaService(
        http_client=AsyncMock(), storage_dir=tmp_path, ocr_engine=ocr
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="image",
        source_url="https://example.test/image.png",
        mime_type="image/png",
        filename="image.png",
        data=b"image",
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="看看图片",
        is_group=True,
        resources=[ResourceMeta(resource_type="image", media_uri=record.media_uri)],
    )

    context = await service.prepare_for_ai(message)
    refreshed = await service.store.authorize("g1", record.media_uri, image_only=True)

    assert "这是截图里的第一行" in context.as_text()
    assert refreshed.summary_model == "PP-OCRv6-small"
    assert ocr.calls  # OCR 被调用了


@pytest.mark.asyncio
async def test_summary_falls_through_to_vlm_when_ocr_finds_no_text(tmp_path):
    multimodal = type("Multimodal", (), {})()
    multimodal.model = "vision-model"
    multimodal.analyze_image = AsyncMock(return_value="画面里有一只猫")
    ocr = FakeOcrEngine([])  # OCR 无文字 → 回退 VLM
    service = MediaService(
        http_client=AsyncMock(),
        multimodal=multimodal,
        storage_dir=tmp_path,
        ocr_engine=ocr,
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="image",
        source_url="https://example.test/image.png",
        mime_type="image/png",
        filename="image.png",
        data=b"image",
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="看看图片",
        is_group=True,
        resources=[ResourceMeta(resource_type="image", media_uri=record.media_uri)],
    )

    context = await service.prepare_for_ai(message)

    assert "画面里有一只猫" in context.as_text()
    multimodal.analyze_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_inspect_uses_vlm_first_and_skips_ocr(tmp_path):
    multimodal = type("Multimodal", (), {})()
    multimodal.model = "vision-model"
    multimodal.analyze_image = AsyncMock(return_value="图片里的人在跑步")
    ocr = FakeOcrEngine(["水印文字"])  # OCR 有文字也不应覆盖 VLM 答案
    service = MediaService(
        http_client=AsyncMock(),
        multimodal=multimodal,
        storage_dir=tmp_path,
        ocr_engine=ocr,
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="image",
        source_url="https://example.test/image.png",
        mime_type="image/png",
        filename="image.png",
        data=b"image",
    )

    result = await service.inspect_image(
        chat_id="g1", media_uri=record.media_uri, question="这个人在做什么？"
    )

    assert result.analysis == "图片里的人在跑步"
    assert not result.note  # 无降级 note
    assert not ocr.calls  # VLM 成功时 OCR 不执行
    assert result.as_dict() == {
        "media_uri": record.media_uri,
        "analysis": "图片里的人在跑步",
        "cached": False,
    }


@pytest.mark.asyncio
async def test_inspect_falls_back_to_ocr_when_vlm_fails(tmp_path):
    multimodal = type("Multimodal", (), {})()
    multimodal.model = "vision-model"
    multimodal.analyze_image = AsyncMock(side_effect=RuntimeError("VLM 挂了"))
    ocr = FakeOcrEngine(["验证码: A1B2", "请输入结果"])
    service = MediaService(
        http_client=AsyncMock(),
        multimodal=multimodal,
        storage_dir=tmp_path,
        ocr_engine=ocr,
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="image",
        source_url="https://example.test/image.png",
        mime_type="image/png",
        filename="image.png",
        data=b"image",
    )

    result = await service.inspect_image(
        chat_id="g1", media_uri=record.media_uri, question="验证码是多少？"
    )

    assert result.analysis == "验证码: A1B2\n请输入结果"
    assert "视觉模型暂不可用" in result.note
    assert result.as_dict()["note"] == result.note
    assert ocr.calls  # OCR 兜底被调用


@pytest.mark.asyncio
async def test_inspect_fails_honestly_when_both_unavailable(tmp_path):
    multimodal = type("Multimodal", (), {})()
    multimodal.model = "vision-model"
    multimodal.analyze_image = AsyncMock(side_effect=RuntimeError("VLM 挂了"))
    ocr = FakeOcrEngine([])  # OCR 也无文字
    service = MediaService(
        http_client=AsyncMock(),
        multimodal=multimodal,
        storage_dir=tmp_path,
        ocr_engine=ocr,
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="image",
        source_url="https://example.test/image.png",
        mime_type="image/png",
        filename="image.png",
        data=b"image",
    )

    result = await service.inspect_image(
        chat_id="g1", media_uri=record.media_uri, question="这是什么？"
    )

    assert result.error == "ANALYSIS_FAILED"


@pytest.mark.asyncio
async def test_image_tools_hidden_when_ocr_unavailable_and_no_vlm(
    tmp_path, monkeypatch
):
    # rapidocr 不可导入 且 无 VLM → image 工具应保持隐藏（不暴露恒失败的工具）
    monkeypatch.setattr("core.media.service.is_ocr_available", lambda: False)
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    assert service.image_tools_enabled is False
    assert service.image_capability.providers == ()
    assert service.image_inspect_capability.providers == ()


@pytest.mark.asyncio
async def test_ocr_disabled_via_config(tmp_path):
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        image_understanding={"ocr": {"enabled": False}},
    )
    assert service.ocr_enabled is False
    assert service.image_capability.providers == ()
    assert service.image_inspect_capability.providers == ()


@pytest.mark.asyncio
async def test_injected_ocr_engine_used_even_when_rapidocr_unavailable(
    tmp_path, monkeypatch
):
    # 显式注入 ocr_engine 时，即使 rapidocr 不可导入也挂载 OCR provider
    monkeypatch.setattr("core.media.service.is_ocr_available", lambda: False)
    ocr = FakeOcrEngine(["这是注入引擎的识别文本内容"])
    service = MediaService(
        http_client=AsyncMock(), storage_dir=tmp_path, ocr_engine=ocr
    )
    assert [p.name for p in service.image_capability.providers] == ["rapidocr"]
    assert service.image_tools_enabled is True
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="image",
        source_url="https://example.test/image.png",
        mime_type="image/png",
        filename="image.png",
        data=b"image",
    )
    result = await service.inspect_image(
        chat_id="g1", media_uri=record.media_uri, question="写了什么？"
    )
    assert result.analysis == "这是注入引擎的识别文本内容"


@pytest.mark.asyncio
async def test_voice_transcription_preloads_caches_and_isolates_sessions(tmp_path):
    transcriber = FakeTranscriber()
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        voice_transcriber=transcriber,
        voice_transcription={"enabled": True},
    )
    await service.open()
    transcriber.preload.assert_awaited_once()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="voice",
        source_url="https://example.test/voice.ogg",
        mime_type="audio/ogg",
        filename="voice.ogg",
        data=b"voice",
    )
    first = await service.transcribe_voice(chat_id="g1", media_uri=record.media_uri)
    second = await service.transcribe_voice(chat_id="g1", media_uri=record.media_uri)
    other_chat = await service.transcribe_voice(
        chat_id="g2", media_uri=record.media_uri
    )
    assert first.transcript == "你好 hello"
    assert not first.cached
    assert second.cached
    transcriber.transcribe.assert_awaited_once()
    assert other_chat.error == "MEDIA_FORBIDDEN"


@pytest.mark.asyncio
async def test_voice_transcription_preserves_voice_type_without_audio_mime(tmp_path):
    transcriber = FakeTranscriber()
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        voice_transcriber=transcriber,
        voice_transcription={"enabled": True},
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="voice",
        source_url="https://example.test/voice",
        mime_type="application/octet-stream",
        filename="voice",
        data=b"voice",
    )

    result = await service.transcribe_voice(chat_id="g1", media_uri=record.media_uri)

    assert result.transcript == "你好 hello"
    transcriber.transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_voice_transcription_single_flight_for_concurrent_requests(tmp_path):
    transcriber = FakeTranscriber()
    started = asyncio.Event()
    release = asyncio.Event()

    async def transcribe(_):
        started.set()
        await release.wait()
        return "你好 hello"

    transcriber.transcribe.side_effect = transcribe
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        voice_transcriber=transcriber,
        voice_transcription={"enabled": True},
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="voice",
        source_url="https://example.test/voice.ogg",
        mime_type="audio/ogg",
        filename="voice.ogg",
        data=b"voice",
    )
    first = asyncio.create_task(
        service.transcribe_voice(chat_id="g1", media_uri=record.media_uri)
    )
    await started.wait()
    second = asyncio.create_task(
        service.transcribe_voice(chat_id="g1", media_uri=record.media_uri)
    )
    release.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.transcript == second_result.transcript == "你好 hello"
    transcriber.transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_media_store_migrates_legacy_index_with_resource_type(tmp_path):
    index_path = tmp_path / "index.sqlite3"
    with sqlite3.connect(index_path) as conn:
        conn.execute(
            "CREATE TABLE media_messages ("
            "chat_id TEXT NOT NULL, message_id TEXT NOT NULL, "
            "sender_id TEXT NOT NULL, media_id TEXT NOT NULL, "
            "position INTEGER NOT NULL, created_at REAL NOT NULL, "
            "PRIMARY KEY (chat_id, message_id, media_id)"
            ")"
        )

    store = MediaStore(tmp_path)
    await store.open()
    columns = {
        row["name"]
        for row in store._conn.execute("PRAGMA table_info(media_messages)").fetchall()
    }

    assert "resource_type" in columns


@pytest.mark.asyncio
async def test_prepare_for_ai_includes_current_voice_transcription(tmp_path):
    transcriber = FakeTranscriber()
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        voice_transcriber=transcriber,
        voice_transcription={"enabled": True},
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="voice",
        source_url="https://example.test/voice.ogg",
        mime_type="audio/ogg",
        filename="voice.ogg",
        data=b"voice",
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="听听这个",
        is_group=True,
        resources=[ResourceMeta(resource_type="voice", media_uri=record.media_uri)],
    )
    text = (await service.prepare_for_ai(message)).as_text()
    assert "[当前语音]" in text
    assert "你好 hello" in text


@pytest.mark.asyncio
async def test_failed_download_does_not_retain_source_url(tmp_path):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    resource = ResourceMeta(
        resource_type="image",
        resource_id="https://example.test/image.jpg",
        source_url="https://example.test/image.jpg",
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="",
        is_group=True,
        resources=[resource],
    )
    service._download = AsyncMock(side_effect=ValueError("failed"))
    await service.ingest_message(message)
    assert resource.storage_status == "failed"
    assert resource.source_url == ""
    assert resource.resource_id == ""
