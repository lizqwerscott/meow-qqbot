"""TTS 语音合成服务

封装 llama-tts-server (VoxCPM/VoxCPM2) 的 OpenAI 兼容接口。
流程：
1. 配置 base_url + 可选 ref_audio（参考音频，启动时自动 base64 编码）
2. synthesize(text, instructions, voice_mode) 合成语音
   - instructions 会自动预置到文本前作为 (instructions) 语音设计前缀
   - voice_mode=preset 使用克隆音色，voice_mode=creative 自由创造
"""

import asyncio
import base64
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

_log = logging.getLogger(__name__)


class TtsService:
    """TTS 语音合成服务 (VoxCPM2 backend)"""

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
    ):
        self._base_url = base_url.rstrip("/")
        self._http = http_client
        self._model = model
        self._temp_dir = Path(temp_dir)

        self._ref_audio_b64: Optional[str] = None
        self._ref_text = ref_text

        self._cfg_value = cfg_value
        self._inference_timesteps = inference_timesteps
        self._temperature = temperature
        self._seed = seed
        self._max_steps = max_steps

        self._temp_dir.mkdir(parents=True, exist_ok=True)

        # 如果配置了参考音频，启动时 base64 编码
        if ref_audio:
            self._encode_ref_audio(ref_audio)

    def _encode_ref_audio(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            _log.warning("参考音频文件不存在: %s", path)
            return
        raw = p.read_bytes()
        self._ref_audio_b64 = base64.b64encode(raw).decode("ascii")
        _log.info("参考音频已加载: %s (%d bytes)", path, len(raw))

    async def synthesize(
        self,
        text: str,
        instructions: Optional[str] = None,
        voice_mode: str = "preset",
    ) -> Optional[bytes]:
        """合成语音

        Args:
            text: 要朗读的文字
            instructions: 说话风格/语气描述（可选），会自动预置到文本前
            voice_mode: "preset" 使用预设克隆音色，"creative" 自由创造音色

        Returns:
            WAV 音频字节，失败返回 None
        """
        # 短文本保护：少于 5 个字时自动补自然填充避免生成断裂
        text = text.strip()
        if len(text) < 5:
            text = text + "……  "

        # 将 instructions 预置为语音设计前缀
        input_text = text
        if instructions:
            input_text = f"({instructions}){text}"

        payload: Dict[str, Any] = {
            "model": self._model,
            "input": input_text,
            "voice": "default",
            "response_format": "wav",
        }
        # preset 模式且配置了参考音频时才传 reference_audio
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
            resp = await self._http.post(
                f"{self._base_url}/v1/audio/speech",
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.content

        except httpx.HTTPStatusError as e:
            _log.error(
                "TTS 合成失败 [%s]: %s",
                e.response.status_code,
                e.response.text[:200],
            )
        except httpx.TimeoutException:
            _log.error("TTS 合成超时")
        except Exception as e:
            _log.error("TTS 合成异常: %s", e, exc_info=True)

        return None

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
