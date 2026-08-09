"""RapidOCR 本地文字识别（PP-OCRv6 small，onnxruntime，无 torch）。

作为图片理解链路的一级：
- 自动摘要：OCR 优先，提取到足够文字就直接当摘要（省 VLM 费用）；
- image 工具：VLM 不可用时回退到 OCR，至少能给出图片中的文字。

rapidocr 是可选依赖，未安装或初始化失败时 OcrEngine 记录失败原因并抛错，
由 MediaCapability 的 provider 链式回退兜底，不影响原有 VLM 路径。
"""

import asyncio
import logging
from typing import Any, List, Optional

_log = logging.getLogger(__name__)

try:  # pragma: no cover - 依赖可选安装
    from rapidocr import RapidOCR as _RapidOcrCls
except Exception as exc:  # pragma: no cover
    _RapidOcrCls = None
    _IMPORT_ERROR: Optional[BaseException] = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


def is_ocr_available() -> bool:
    """rapidocr 是否可导入；未安装时返回 False，用于决定是否挂载 OCR provider。"""
    return _RapidOcrCls is not None


class OcrEngine:
    """懒加载、线程内推理的 RapidOCR 引擎（进程内共享单例）。"""

    def __init__(self) -> None:
        self._engine: Any = None
        self._init_task: Optional[asyncio.Task] = None
        self._init_failed: Optional[BaseException] = None

    async def initialize(self) -> None:
        """首次调用时才加载模型；失败后记录原因，后续直接短路。

        初始化在缓存的后台任务中执行并用 shield 保护：即使某个调用方被取消
        （如 MediaCapability 超时），初始化仍会完成，后续调用直接复用就绪的
        引擎，不会重复加载模型。
        """
        if self._engine is not None or self._init_failed is not None:
            return
        if self._init_task is None:
            self._init_task = asyncio.create_task(self._do_init())
        try:
            await asyncio.shield(self._init_task)
        except asyncio.CancelledError:
            if self._init_task.cancelled():
                # 初始化任务自身被取消（罕见，如事件循环关闭）：重置以便下次重试
                self._init_task = None
            raise

    async def _do_init(self) -> None:
        if _RapidOcrCls is None:
            self._init_failed = _IMPORT_ERROR or ImportError(
                "rapidocr 未安装，请执行 uv add rapidocr onnxruntime"
            )
            _log.warning("RapidOCR 不可用: %s", self._init_failed)
            return
        try:
            # 构造时解析配置/校验模型文件，放线程避免阻塞事件循环
            self._engine = await asyncio.to_thread(_RapidOcrCls)
            _log.info("RapidOCR 引擎就绪（PP-OCRv6 small）")
        except Exception as exc:
            self._init_failed = exc
            _log.warning("RapidOCR 初始化失败: %s", exc)

    async def recognize(self, image_path: str) -> List[str]:
        """返回识别的文本行（原始内容，不做清洗）；无文字时返回空列表。"""
        await self.initialize()
        if self._engine is None:
            raise RuntimeError(f"RapidOCR 不可用: {self._init_failed}")
        result = await asyncio.to_thread(self._engine, image_path)
        # rapidocr>=3 直接返回 RapidOCROutput；兼容旧版 (result, elapse) 元组
        if isinstance(result, tuple) and result:
            result = result[0]
        txts = getattr(result, "txts", None)
        if not txts:
            return []
        return [str(line) for line in txts]


class OcrProvider:
    """实现 MediaCapabilityProvider 协议，把 OCR 文本作为分析结果返回。

    min_chars：有效文本长度下限，低于该值返回空串，链式回退到下一 provider
    （避免把只有零星噪点文字的照片当成“有文字”而跳过 VLM）。
    """

    name = "rapidocr"
    model_name = "PP-OCRv6-small"

    def __init__(
        self,
        engine: Optional[OcrEngine] = None,
        *,
        min_chars: int = 8,
        max_chars: int = 2000,
    ) -> None:
        self._engine = engine or OcrEngine()
        self.min_chars = max(0, int(min_chars))
        self.max_chars = max(1, int(max_chars))

    async def execute(self, record: Any, **kwargs: Any) -> str:
        lines = await self._engine.recognize(str(record.local_path))
        # 归一化单点负责：清洗 + 空行过滤 + 真实字符数判定 + 截断
        lines = [line.strip() for line in lines if line and line.strip()]
        # min_chars 按真实字符计数（不含换行），先判定再截断
        if sum(len(line) for line in lines) < self.min_chars:
            return ""
        text = "\n".join(lines)
        return text[: self.max_chars]
