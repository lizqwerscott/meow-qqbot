"""TTS 语音合成服务

封装 Qwen3-TTS 的 OpenAI 兼容接口。
流程：
1. 配置 voice（已有音色名）或 ref_audio（参考音频路径，启动时自动上传）
2. initialize() 初始化音色
3. synthesize(text, instructions) 合成语音，始终使用 voice + instructions 的 CustomVoice 模式
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

_log = logging.getLogger(__name__)


class TtsService:
    """TTS 语音合成服务"""

    def __init__(
        self,
        base_url: str,
        http_client: httpx.AsyncClient,
        voices_file: str = "data/tts_voices.json",
        temp_dir: str = "data/tts_temp/",
        normalize: Optional[bool] = None,
        cfg_value: Optional[float] = None,
        inference_timesteps: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._http = http_client
        self._voices_file = Path(voices_file)
        self._temp_dir = Path(temp_dir)

        self._voice_name: Optional[str] = None
        self._ref_audio: Optional[str] = None
        self._ref_text: Optional[str] = None

        self._normalize = normalize
        self._cfg_value = cfg_value
        self._inference_timesteps = inference_timesteps
        self._temperature = temperature
        self._seed = seed

        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._voices_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._voices_file.exists():
            self._voices_file.write_text("[]", encoding="utf-8")

    # ── 配置 ──

    def configure(
        self,
        voice: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
    ) -> None:
        """配置音色

        Args:
            voice: 音色名。如果有 ref_audio，会用这个名字上传注册
            ref_audio: 参考音频路径（首次使用，上传后该音频注册为 voice 这个名字）
            ref_text: 参考音频的文字稿（可选，用于更好克隆质量）
        """
        self._voice_name = voice
        self._ref_audio = ref_audio
        self._ref_text = ref_text

    async def initialize(self) -> str:
        """初始化音色。

        如果有 ref_audio，上传参考音频并注册为配置的 voice 名字。
        如果只有 voice（无 ref_audio），直接使用已有音色。
        """
        if self._ref_audio:
            if not self._voice_name:
                raise RuntimeError(
                    "设置了 ref_audio 但未指定 voice 名字，"
                    "请在配置中加上 voice = \"你的音色名\""
                )
            result = await self._upload_voice_file(
                audio_path=self._ref_audio,
                name=self._voice_name,
                consent="auto",
                ref_text=self._ref_text,
            )
            self._save_voice(self._voice_name)
            _log.info("TTS 音色已上传并注册: voice=%s", self._voice_name)
            return self._voice_name

        if not self._voice_name:
            raise RuntimeError(
                "TTS 未配置: 请在 config.toml 中设置 voice（音色名）"
                " 或加上 ref_audio（参考音频路径，首次使用）"
            )

        _log.info("TTS 使用已有音色: %s", self._voice_name)
        return self._voice_name

    # ── 合成（始终 CustomVoice 模式：voice + 可选 instructions） ──

    async def synthesize(
        self,
        text: str,
        instructions: Optional[str] = None,
    ) -> Optional[bytes]:
        """合成语音

        Args:
            text: 要朗读的文字
            instructions: 说话风格/语气（可选），如'热情地欢呼'、'温柔地慢慢说'

        Returns:
            WAV 音频字节，失败返回 None
        """
        if not self._voice_name:
            _log.error("TTS 音色未初始化")
            return None

        payload: Dict[str, Any] = {
            "input": text,
            "voice": self._voice_name,
        }
        if instructions:
            payload["instructions"] = instructions
        if self._normalize is not None:
            payload["normalize"] = self._normalize
        if self._cfg_value is not None:
            payload["cfg_value"] = self._cfg_value
        if self._inference_timesteps is not None:
            payload["inference_timesteps"] = self._inference_timesteps
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._seed is not None:
            payload["seed"] = self._seed

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

    # ── 语音上传 ──

    async def upload_voice(
        self,
        audio_path: str,
        name: str,
        consent: str = "auto",
        ref_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """上传参考音频到 TTS 服务器并保存到本地记录"""
        result = await self._upload_voice_file(audio_path, name, consent, ref_text)
        self._save_voice(name)
        return result

    async def _upload_voice_file(
        self,
        audio_path: str,
        name: str,
        consent: str,
        ref_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"参考音频不存在: {audio_path}")

        files: Dict[str, Any] = {
            "audio_sample": (path.name, path.read_bytes(), "audio/wav"),
            "consent": (None, consent),
            "name": (None, name),
        }
        if ref_text:
            files["ref_text"] = (None, ref_text)

        try:
            resp = await self._http.post(
                f"{self._base_url}/v1/audio/voices",
                files=files,
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            _log.info("语音上传成功: name=%s", name)
            return data
        except Exception as e:
            _log.error("语音上传失败: %s", e, exc_info=True)
            raise

    # ── 本地语音管理 ──

    def list_voices(self) -> List[Dict[str, Any]]:
        """列出已保存的语音"""
        try:
            return json.loads(self._voices_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def delete_voice(self, name: str) -> bool:
        """删除本地保存的语音记录"""
        voices = self.list_voices()
        before = len(voices)
        voices = [v for v in voices if v.get("name") != name]
        if len(voices) == before:
            return False
        self._voices_file.write_text(
            json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True

    def _save_voice(self, name: str) -> None:
        voices = self.list_voices()
        # 去重
        voices = [v for v in voices if v.get("name") != name]
        voices.append({
            "name": name,
            "created_at": time.time(),
        })
        self._voices_file.write_text(
            json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── 临时文件管理 ──

    def save_temp_audio(self, data: bytes) -> str:
        """保存临时音频文件，返回路径"""
        filename = f"tts_{uuid.uuid4().hex}.wav"
        path = self._temp_dir / filename
        path.write_bytes(data)
        return str(path)

    def cleanup_temp(self, age_hours: int = 1) -> int:
        """清理过期的临时文件"""
        now = time.time()
        count = 0
        for f in self._temp_dir.glob("tts_*.wav"):
            if now - f.stat().st_mtime > age_hours * 3600:
                f.unlink(missing_ok=True)
                count += 1
        return count

    async def close(self) -> None:
        self.cleanup_temp(0)
