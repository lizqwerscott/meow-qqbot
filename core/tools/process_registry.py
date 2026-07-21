"""ProcessRegistry — 后台进程注册表（in-memory）。

管理通过 exec 工具 background 模式启动的进程会话。
支持输出缓冲、stdin 写入、终止、TTL 清理。
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

_log = logging.getLogger(__name__)


@dataclass
class ProcessSession:
    id: str
    command: str
    workdir: Optional[str] = None
    chat_id: str = ""
    delivery_channel: Optional[str] = None
    created_at: float = 0.0
    pid: Optional[int] = None
    process: Optional[asyncio.subprocess.Process] = None
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)
    stdout_done: bool = False
    stderr_done: bool = False
    exited: bool = False
    exit_code: Optional[int] = None
    backgrounded: bool = False
    _read_task: Optional[asyncio.Task] = None


class ProcessRegistry:
    """In-memory registry for background process sessions.

    用法:
        registry = ProcessRegistry()
        registry.on_exit(my_callback)
        await registry.start()
        session_id = await registry.spawn(command, parts)
        # ... 使用 process 工具管理 ...
        await registry.stop()
    """

    DEFAULT_TTL = 1800

    def __init__(self, ttl: int = DEFAULT_TTL):
        self._sessions: dict[str, ProcessSession] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl
        self._cleanup_task: Optional[asyncio.Task] = None
        self._exit_callbacks: list[Callable] = []

    def on_exit(self, callback: Callable) -> None:
        """注册进程退出回调。回调签名: async (session: ProcessSession) -> None"""
        self._exit_callbacks.append(callback)

    async def start(self) -> None:
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        async with self._lock:
            for session in list(self._sessions.values()):
                self._kill_session(session)
            self._sessions.clear()

    async def spawn(
        self,
        command: str,
        parts: list[str],
        workdir: Optional[str] = None,
        chat_id: str = "",
        delivery_channel: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        session_id = uuid.uuid4().hex[:16]
        session = ProcessSession(
            id=session_id,
            command=command,
            workdir=workdir,
            chat_id=chat_id,
            delivery_channel=delivery_channel,
            created_at=time.time(),
            backgrounded=True,
        )
        async with self._lock:
            self._sessions[session_id] = session

        try:
            process = await asyncio.create_subprocess_exec(
                *parts,
                cwd=workdir or ".",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE,
            )
            session.pid = process.pid
            session.process = process
            session._read_task = asyncio.create_task(
                self._read_streams(session)
            )
            _log.info(
                "后台进程已启动 [%s..]: pid=%d cmd=%s",
                session_id[:8], process.pid, command[:80],
            )
        except Exception as e:
            async with self._lock:
                self._sessions.pop(session_id, None)
            _log.warning("后台进程启动失败 [%s..]: %s", session_id[:8], e)
            raise

        return session_id

    async def _read_streams(self, session: ProcessSession) -> None:
        proc = session.process
        if not proc:
            session.exited = True
            session.exit_code = -1
            return
        try:
            stdout_task = asyncio.create_task(
                self._read_stream(proc.stdout, session.stdout_lines)
            )
            stderr_task = asyncio.create_task(
                self._read_stream(proc.stderr, session.stderr_lines)
            )
            await asyncio.gather(stdout_task, stderr_task)
            exit_code = await proc.wait()
            session.exited = True
            session.exit_code = exit_code
            _log.info(
                "后台进程已退出 [%s..]: pid=%d exit=%d",
                session.id[:8], session.pid, exit_code,
            )
            for cb in self._exit_callbacks:
                try:
                    await cb(session)
                except Exception:
                    _log.exception("进程退出回调异常 [%s..]", session.id[:8])
        except asyncio.CancelledError:
            pass
        except Exception:
            _log.exception("进程读取异常 [%s..]", session.id[:8])

    async def _read_stream(self, stream, lines_list: list[str]) -> None:
        if not stream:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip("\r\n")
                lines_list.append(decoded)
        except Exception:
            pass

    async def write_stdin(self, session_id: str, data: str) -> Optional[str]:
        session = await self._get(session_id)
        if not session:
            return "会话不存在"
        if session.exited:
            return "进程已退出"
        if not session.process or not session.process.stdin:
            return "stdin 不可用"
        try:
            session.process.stdin.write(data.encode("utf-8"))
            await session.process.stdin.drain()
            return None
        except Exception as e:
            return f"写入 stdin 失败: {e}"

    async def kill(self, session_id: str) -> Optional[str]:
        session = await self._get(session_id)
        if not session:
            return "会话不存在"
        if session.exited:
            return None
        self._kill_session(session)
        return None

    async def remove(self, session_id: str) -> Optional[str]:
        session = await self._get(session_id)
        if not session:
            return "会话不存在"
        if session._read_task and not session._read_task.done():
            session._read_task.cancel()
        self._kill_session(session)
        async with self._lock:
            self._sessions.pop(session_id, None)
        return None

    def _kill_session(self, session: ProcessSession) -> None:
        if session.process and session.process.returncode is None:
            try:
                session.process.terminate()
            except ProcessLookupError:
                pass

    async def list_sessions(self) -> list[dict]:
        async with self._lock:
            return [self._summary(s) for s in self._sessions.values()]

    async def get_log(
        self, session_id: str, offset: int = 0, limit: int = 200
    ) -> Optional[dict]:
        session = await self._get(session_id)
        if not session:
            return None

        total_stdout = len(session.stdout_lines)
        total_stderr = len(session.stderr_lines)

        if limit is not None and offset == 0:
            start = max(total_stdout - limit, 0)
            stdout_slice = session.stdout_lines[start:]
        else:
            stdout_slice = session.stdout_lines[offset:offset + limit]

        tail_note = ""
        if limit and total_stdout > limit and offset == 0:
            tail_note = (
                f"\n[仅显示最后 {limit}/{total_stdout} 行 stdout; "
                f"使用 offset/limit 参数分页查看]"
            )

        return {
            "stdout": "\n".join(stdout_slice) + tail_note,
            "stderr": "\n".join(session.stderr_lines),
            "total_stdout": total_stdout,
            "total_stderr": total_stderr,
            "offset": offset,
            "limit": limit or total_stdout,
        }

    async def poll(
        self, session_id: str, timeout: float = 30.0
    ) -> Optional[dict]:
        session = await self._get(session_id)
        if not session:
            return None

        if session.exited:
            return self._summary(session)

        if timeout > 0 and not session.stdout_done and not session.stderr_done:
            await asyncio.sleep(min(timeout, 0.5))

        return self._summary(session)

    async def _get(self, session_id: str) -> Optional[ProcessSession]:
        async with self._lock:
            return self._sessions.get(session_id)

    def _summary(self, session: ProcessSession) -> dict:
        return {
            "session_id": session.id,
            "command": session.command,
            "pid": session.pid,
            "created_at": session.created_at,
            "exited": session.exited,
            "exit_code": session.exit_code,
            "backgrounded": session.backgrounded,
            "stdout_lines": len(session.stdout_lines),
            "stderr_lines": len(session.stderr_lines),
        }

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(300)
            now = time.time()
            async with self._lock:
                expired = [
                    sid for sid, s in list(self._sessions.items())
                    if s.exited and (now - s.created_at) > self._ttl
                ]
                for sid in expired:
                    s = self._sessions.pop(sid)
                    if s._read_task and not s._read_task.done():
                        s._read_task.cancel()
                if expired:
                    _log.info("清理了 %d 个过期进程会话", len(expired))
