"""TTS 语音合成服务。

支持两种后端：
- ``voxcpm``：llama-tts-server 的 OpenAI 兼容接口；
- ``s2-pro``：Fish Audio S2 Pro server 的 ``/generate`` multipart 接口。
"""

import asyncio
import base64
import io
import json
import logging
import mimetypes
import struct
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import numpy as np

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TtsBackendConfig:
    """后端模型的接口与工具提示配置。"""

    text_rules: str
    instructions_rules: str
    supports_instructions: bool
    voice_mode_rules: str
    tag_examples: str
    voice_modes: tuple[str, ...]
    default_base_url: str


class TtsService:
    """TTS 语音合成服务。"""

    _BACKEND_CONFIGS = {
        "voxcpm": TtsBackendConfig(
            text_rules=(
                "VoxCPM：使用短句和标点控制停顿（句号长停顿、逗号短停顿、省略号表示犹豫）。"
                "需要非语言表现时，可直接在正文中使用 [laughing]、[sigh] 等标签。"
            ),
            instructions_rules=(
                "VoxCPM：描述语速、情绪和表达方式；服务会自动将其包进 (...) 后置于正文前。"
                "preset 只描述情绪/语速/表达，不要改变音色身份；creative 可完整描述身份、音色和情绪。"
            ),
            supports_instructions=True,
            voice_mode_rules=(
                "VoxCPM：preset（默认）使用管理员预设的克隆音色；creative 不使用预设音色，"
                "可自由设计新声音。"
            ),
            tag_examples="",
            voice_modes=("preset", "creative"),
            default_base_url="http://localhost:8080",
        ),
        "s2-pro": TtsBackendConfig(
            text_rules=(
                "S2-Pro：情绪、语气和效果标签必须使用 [bracket] 语法，并放在所修饰句子之前，"
                "例如 [excited] 今天真不错。标签可用自然语言描述，但每句最多一个且不要滥用。"
                "正文使用带句号的自然句；长文本使用干净标点也可正常生成；禁止分号和圆括号。"
            ),
            instructions_rules="",
            supports_instructions=False,
            voice_mode_rules=(
                "S2-Pro：仅支持 preset（默认）；管理员同时配置 ref_audio 和 ref_text 时使用克隆音色，"
                "未配置参考音频时按服务默认音色生成。"
            ),
            tag_examples=(
                "S2-Pro 标签示例（支持自由自然语言描述）：\n"
                "- 情绪：[excited] [curious] [skeptical] [frustrated] [delighted] [grateful] "
                "[confident] [sad] [angry] [surprised]\n"
                "- 语气：[whispering] [shouting] [soft tone] [in a hurry tone]\n"
                "- 效果：[laughing] [chuckling] [sighing] [sobbing] [panting]\n"
                "- 自然描述：[speaking slowly and solemnly] [with deep sincerity] "
                "[with quiet confidence]\n"
                "标签放在所修饰句子前；每句最多一个标签；短文本不要堆叠标签；"
                "相邻句子可使用对比情绪增加表现力。"
            ),
            voice_modes=("preset",),
            default_base_url="http://localhost:3030",
        ),
    }
    _SUPPORTED_BACKENDS = set(_BACKEND_CONFIGS)

    def __init__(
        self,
        base_url: str,
        http_client: httpx.AsyncClient,
        model: str = "voxcpm",
        temp_dir: str = "data/tts_temp/",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        cfg_value: Optional[float] = None,
        inference_timesteps: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        backend: str = "voxcpm",
        s2_params: Optional[Dict[str, Any]] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._http = http_client
        self._model = model
        self._temp_dir = Path(temp_dir)
        self._backend, self._backend_config = self.resolve_backend(backend)

        self._ref_audio_b64: Optional[str] = None
        self._ref_audio_data: Optional[bytes] = None
        self._ref_audio_name: Optional[str] = None
        self._ref_audio_mime: Optional[str] = None
        self._ref_text = ref_text.strip() if ref_text else None
        self._s2_params = dict(s2_params or {})

        self._cfg_value = cfg_value
        self._inference_timesteps = inference_timesteps
        self._temperature = temperature
        self._seed = seed
        self._max_steps = max_steps

        self._temp_dir.mkdir(parents=True, exist_ok=True)

        if ref_audio:
            self._load_ref_audio(ref_audio)
        if self._backend == "s2-pro" and self._ref_audio_data and not self._ref_text:
            _log.warning("S2-Pro 已配置参考音频，但缺少 ref_text；将不使用音色克隆")

    @classmethod
    def resolve_backend(cls, backend: str) -> tuple[str, TtsBackendConfig]:
        """规范化后端名并返回其配置。"""
        normalized_backend = backend.lower().replace("_", "-")
        backend_config = cls._BACKEND_CONFIGS.get(normalized_backend)
        if backend_config is None:
            raise ValueError(
                f"不支持的 TTS backend: {backend}，可选值: {', '.join(sorted(cls._SUPPORTED_BACKENDS))}"
            )
        return normalized_backend, backend_config

    @property
    def tool_config(self) -> TtsBackendConfig:
        """返回当前后端应注入到 TTS 工具定义的模型规则。"""
        return self._backend_config

    @classmethod
    def default_tool_config(cls) -> TtsBackendConfig:
        """TTS 未启用时，工具定义使用的向后兼容默认规则。"""
        return cls._BACKEND_CONFIGS["voxcpm"]

    def _load_ref_audio(self, path: str) -> None:
        ref_path = Path(path)
        if not ref_path.is_file():
            _log.warning("参考音频文件不存在: %s", path)
            return

        raw = ref_path.read_bytes()
        self._ref_audio_b64 = base64.b64encode(raw).decode("ascii")
        self._ref_audio_data = raw
        self._ref_audio_name = ref_path.name
        self._ref_audio_mime = mimetypes.guess_type(ref_path.name)[0] or "audio/wav"
        _log.info("参考音频已加载: %s (%d bytes)", path, len(raw))

    async def synthesize(
        self,
        text: str,
        instructions: Optional[str] = None,
        voice_mode: str = "preset",
    ) -> Optional[bytes]:
        """合成语音，失败返回 ``None``。"""
        text = text.strip()
        if len(text) < 5:
            text += "……  "

        if voice_mode not in self.tool_config.voice_modes:
            _log.warning(
                "TTS backend=%s 不支持 voice_mode=%s", self._backend, voice_mode
            )
            return None

        if instructions and not self.tool_config.supports_instructions:
            _log.warning(
                "TTS backend=%s 不支持 instructions；请将方括号标签直接写入 text",
                self._backend,
            )
            return None

        if self._backend == "s2-pro":
            return await self._synthesize_s2_pro(self._normalize_s2_text(text))
        return await self._synthesize_voxcpm(text, instructions, voice_mode)

    async def _synthesize_voxcpm(
        self,
        text: str,
        instructions: Optional[str],
        voice_mode: str,
    ) -> Optional[bytes]:
        input_text = f"({instructions}){text}" if instructions else text
        payload: Dict[str, Any] = {
            "model": self._model,
            "input": input_text,
            "voice": "default",
            "response_format": "wav",
        }
        if voice_mode == "preset" and self._ref_audio_b64:
            payload["reference_audio"] = self._ref_audio_b64
        if self._cfg_value is not None:
            payload["cfg_value"] = self._cfg_value
        if self._inference_timesteps is not None:
            payload["inference_timesteps"] = self._inference_timesteps
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._seed is not None:
            payload["seed"] = self._seed
        if self._max_steps is not None:
            payload["max_steps"] = self._max_steps

        try:
            response = await self._http.post(
                f"{self._base_url}/v1/audio/speech",
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            _log.error(
                "TTS 合成失败 [%s]: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
        except httpx.TimeoutException:
            _log.error("TTS 合成超时")
        except Exception as exc:
            _log.error("TTS 合成异常: %s", exc, exc_info=True)
        return None

    @staticmethod
    def _normalize_s2_text(value: str) -> str:
        """替换 S2-Pro 不支持的标点，同时保留其余文本和方括号标签。"""
        return value.translate(
            str.maketrans(
                {
                    ";": "，",
                    "；": "，",
                    "(": "",
                    ")": "",
                    "（": "",
                    "）": "",
                }
            )
        ).strip()

    async def _synthesize_s2_pro(self, text: str) -> Optional[bytes]:
        files = {"text": (None, text)}
        if self._ref_audio_data and self._ref_text:
            files["reference_text"] = (None, self._ref_text)
            files["reference"] = (
                self._ref_audio_name or "reference.wav",
                self._ref_audio_data,
                self._ref_audio_mime or "audio/wav",
            )
        if self._s2_params:
            files["params"] = (None, json.dumps(self._s2_params, ensure_ascii=False))

        try:
            response = await self._http.post(
                f"{self._base_url}/generate",
                files=files,
                timeout=60.0,
            )
            response.raise_for_status()
            return self._convert_s2_float_wav(response.content)
        except httpx.HTTPStatusError as exc:
            _log.error(
                "S2-Pro 合成失败 [%s]: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
        except httpx.TimeoutException:
            _log.error("S2-Pro 合成超时")
        except Exception as exc:
            _log.error("S2-Pro 合成异常: %s", exc, exc_info=True)
        return None

    @staticmethod
    def _convert_s2_float_wav(data: bytes) -> bytes:
        """将 S2-Pro 的 IEEE-float WAV 转为 QQ 可播放的 16-bit PCM WAV。"""
        if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise ValueError("S2-Pro 响应不是有效的 WAV 文件")

        audio_format = channels = sample_rate = bits_per_sample = None
        audio_data = None
        offset = 12
        while offset + 8 <= len(data):
            chunk_id = data[offset : offset + 4]
            chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
            chunk_start = offset + 8
            chunk_end = chunk_start + chunk_size
            if chunk_end > len(data):
                raise ValueError("WAV chunk 长度无效")
            if chunk_id == b"fmt ":
                if chunk_size < 16:
                    raise ValueError("WAV fmt chunk 无效")
                audio_format, channels, sample_rate, _, _, bits_per_sample = (
                    struct.unpack_from("<HHIIHH", data, chunk_start)
                )
            elif chunk_id == b"data":
                audio_data = data[chunk_start:chunk_end]
            offset = chunk_end + (chunk_size % 2)

        if None in (audio_format, channels, sample_rate, bits_per_sample, audio_data):
            raise ValueError("WAV 缺少 fmt 或 data chunk")
        if audio_format == 1:
            return data
        if audio_format != 3 or bits_per_sample != 32 or len(audio_data) % 4:
            raise ValueError("S2-Pro WAV 不是 32-bit float PCM")

        samples = np.frombuffer(audio_data, dtype="<f4")
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak:
            samples = samples * (0.95 / peak)
        pcm_data = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()

        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_data)
        return output.getvalue()

    # ── 临时文件管理 ──

    def save_temp_audio(self, data: bytes) -> str:
        """保存临时音频文件，返回路径"""
        filename = f"tts_{uuid.uuid4().hex}.wav"
        path = self._temp_dir / filename
        path.write_bytes(data)
        return str(path)

    async def cleanup_temp(self, age_hours: int = 1) -> int:
        """清理过期的临时文件"""
        now = time.time()
        count = 0
        for f in self._temp_dir.glob("tts_*.wav"):
            if now - f.stat().st_mtime > age_hours * 3600:
                await asyncio.to_thread(f.unlink, missing_ok=True)
                count += 1
        return count

    async def close(self) -> None:
        await self.cleanup_temp(0)
