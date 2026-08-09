import asyncio

import pytest

from core.media.ocr import OcrEngine, OcrProvider


class FakeEngine:
    def __init__(self, lines):
        self._lines = lines

    async def recognize(self, image_path):
        return list(self._lines)


class FakeRecord:
    def __init__(self, path="img.png"):
        self.local_path = path


@pytest.mark.asyncio
async def test_ocr_provider_returns_joined_lines():
    provider = OcrProvider(engine=FakeEngine(["第一行文字", "第二行文字"]))
    result = await provider.execute(FakeRecord())
    assert result == "第一行文字\n第二行文字"


@pytest.mark.asyncio
async def test_ocr_provider_empty_when_below_min_chars():
    provider = OcrProvider(engine=FakeEngine(["短"]), min_chars=8)
    result = await provider.execute(FakeRecord())
    assert result == ""


@pytest.mark.asyncio
async def test_ocr_provider_respects_min_chars_boundary():
    provider = OcrProvider(engine=FakeEngine(["一二三四五六七八"]), min_chars=8)
    result = await provider.execute(FakeRecord())
    assert result == "一二三四五六七八"


@pytest.mark.asyncio
async def test_ocr_provider_no_text_returns_empty():
    provider = OcrProvider(engine=FakeEngine([]))
    result = await provider.execute(FakeRecord())
    assert result == ""


@pytest.mark.asyncio
async def test_ocr_provider_truncates_max_chars():
    provider = OcrProvider(engine=FakeEngine(["A" * 100]), max_chars=10)
    result = await provider.execute(FakeRecord())
    assert result == "A" * 10


@pytest.mark.asyncio
async def test_min_chars_counts_real_chars_excluding_newlines():
    # 3 行 "ab" 拼接后含换行为 8 字符，但真实字符仅 6 个
    provider = OcrProvider(engine=FakeEngine(["ab", "ab", "ab"]), min_chars=8)
    assert await provider.execute(FakeRecord()) == ""
    provider = OcrProvider(engine=FakeEngine(["ab", "ab", "ab"]), min_chars=6)
    assert await provider.execute(FakeRecord()) == "ab\nab\nab"


@pytest.mark.asyncio
async def test_max_chars_smaller_than_min_chars_still_returns_text():
    # min 判定基于截断前的真实字符数，max < min 不应导致永远空串
    provider = OcrProvider(
        engine=FakeEngine(["一二三四五六七八"]), min_chars=8, max_chars=5
    )
    result = await provider.execute(FakeRecord())
    assert result == "一二三四五六七"[:5]
    assert len(result) == 5


@pytest.mark.asyncio
async def test_ocr_provider_skips_blank_lines():
    provider = OcrProvider(engine=FakeEngine(["有字", "  ", "", "也有字"]), min_chars=1)
    result = await provider.execute(FakeRecord())
    assert result == "有字\n也有字"


class _BrokenEngine:
    async def recognize(self, image_path):
        raise RuntimeError("rapidocr 不可用")


@pytest.mark.asyncio
async def test_ocr_provider_raises_when_engine_fails():
    provider = OcrProvider(engine=_BrokenEngine())
    with pytest.raises(RuntimeError):
        await provider.execute(FakeRecord())


@pytest.mark.asyncio
async def test_ocr_engine_initializes_once():
    calls = []

    class CountingCls:
        def __init__(self):
            calls.append(1)

    module = __import__("core.media.ocr", fromlist=["x"])
    original = module._RapidOcrCls
    module._RapidOcrCls = CountingCls
    try:
        engine = OcrEngine()
        await engine.initialize()
        await engine.initialize()
        assert len(calls) == 1
    finally:
        module._RapidOcrCls = original


@pytest.mark.asyncio
async def test_ocr_engine_concurrent_initialize_loads_once():
    calls = []

    class CountingCls:
        def __init__(self):
            calls.append(1)

    module = __import__("core.media.ocr", fromlist=["x"])
    original = module._RapidOcrCls
    module._RapidOcrCls = CountingCls
    try:
        engine = OcrEngine()
        await asyncio.gather(
            engine.initialize(), engine.initialize(), engine.initialize()
        )
        assert len(calls) == 1
    finally:
        module._RapidOcrCls = original


@pytest.mark.asyncio
async def test_ocr_engine_failure_short_circuits_without_retry():
    calls = []

    class RaisingCls:
        def __init__(self):
            calls.append(1)
            raise RuntimeError("模型加载失败")

    module = __import__("core.media.ocr", fromlist=["x"])
    original = module._RapidOcrCls
    module._RapidOcrCls = RaisingCls
    try:
        engine = OcrEngine()
        await engine.initialize()
        assert engine._engine is None
        assert engine._init_failed is not None
        await engine.initialize()  # 失败短路：不重试
        assert len(calls) == 1
        with pytest.raises(RuntimeError):
            await engine.recognize("img.png")
    finally:
        module._RapidOcrCls = original


@pytest.mark.asyncio
async def test_ocr_engine_missing_package_path():
    module = __import__("core.media.ocr", fromlist=["x"])
    original_cls, original_err = module._RapidOcrCls, module._IMPORT_ERROR
    module._RapidOcrCls = None
    module._IMPORT_ERROR = ImportError("rapidocr 未安装")
    try:
        engine = OcrEngine()
        await engine.initialize()
        assert engine._init_failed is not None
        with pytest.raises(RuntimeError):
            await engine.recognize("img.png")
    finally:
        module._RapidOcrCls, module._IMPORT_ERROR = original_cls, original_err


@pytest.mark.asyncio
async def test_is_ocr_available(monkeypatch):
    import core.media.ocr as ocr_module

    assert ocr_module.is_ocr_available() is (ocr_module._RapidOcrCls is not None)
    monkeypatch.setattr(ocr_module, "_RapidOcrCls", None)
    assert ocr_module.is_ocr_available() is False
    monkeypatch.setattr(ocr_module, "_RapidOcrCls", object)
    assert ocr_module.is_ocr_available() is True
