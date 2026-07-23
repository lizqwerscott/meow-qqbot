"""TranscriptFilter — span-aware 心跳产物过滤。

与 isolated_session 的分工：
  - isolated_session=True: 不把心跳内容发给 LLM（省钱）
  - TranscriptFilter:      不给用户看心跳轮次（不泄露）

匹配 OpenClaw heartbeat-filter.ts 的 span-aware 状态机。
"""




def is_heartbeat_ack_only(content: str) -> bool:
    """检测仅含 HEARTBEAT_OK / NO_REPLY 的纯静默回复。"""
    from .delivery_normalization import strip_heartbeat_token
    _, should_skip = strip_heartbeat_token(str(content))
    return should_skip


def is_silent_heartbeat_respond(msg: dict) -> bool:
    """检测 heartbeat_respond(notify=false) 工具调用。"""
    for tc in msg.get("tool_calls") or []:
        func = tc.get("function", {})
        if func.get("name") == "heartbeat_respond":
            args = func.get("arguments", "")
            if '"notify": false' in args or '"notify":false' in args.replace(" ", ""):
                return True
    return False


def is_heartbeat_user_msg(content: str) -> bool:
    """6 种模式匹配 heartbeat user 消息。"""
    if not content:
        return False
    for pattern in [
        "以下系统事件需要关注",
        "心跳检查",
        "定时心跳",
        "后台进程完成",
        "【定时任务事件】",
        "【后台进程完成】",
    ]:
        if pattern in content:
            return True
    if content.startswith("[系统提醒]") or content.startswith("[后台通知]"):
        return True
    return False


def filter_heartbeat_artifact_spans(
    messages: list[dict],
    heartbeat_prompt: str = "",
) -> list[dict]:
    """Span-aware 状态机过滤。

    逻辑：
    1. 扫描到 heartbeat user 消息 → 进入 span
    2. 在 span 内：
       - heartbeat_respond(notify=false) → span 静默结束，全部丢弃
       - HEARTBEAT_OK/NO_REPLY ack → 同上
       - 有非静默内容的 assistant → span 保护，保留 assistant + 其 tool 结果
       - tool 结果 → 同 span 内
       - 真实 user 消息 → 退出 span
    3. 非 span 内消息全部保留
    """
    filtered: list[dict] = []
    in_span = False
    protect_span = False
    skip_tool_count = 0  # 跳过助理静默响应后的 tool 结果

    for msg in messages:
        role = msg.get("role", "")

        # 跳过跟随在静默 assistant 之后的 tool 结果
        if skip_tool_count > 0:
            if role == "tool":
                skip_tool_count -= 1
                continue
            skip_tool_count = 0  # 非 tool → 正常处理

        if not in_span and role == "user" and is_heartbeat_user_msg(str(msg.get("content", ""))):
            in_span = True
            protect_span = False
            continue

        if in_span:
            if role == "assistant":
                if is_silent_heartbeat_respond(msg) or is_heartbeat_ack_only(
                    str(msg.get("content", ""))
                ):
                    # 静默 span → 全部丢弃，跳过随后的 tool 结果
                    in_span = False
                    skip_tool_count = 2  # 跳过最多 2 条 tool 结果
                    continue
                content = str(msg.get("content", ""))
                if content.strip():
                    # 有非静默内容 → span 保护
                    protect_span = True
                    in_span = False
                    filtered.append(msg)
                    continue
                continue
            elif role == "tool":
                if protect_span:
                    filtered.append(msg)
                continue
            elif role == "user":
                if is_heartbeat_user_msg(str(msg.get("content", ""))):
                    continue
                in_span = False
                filtered.append(msg)
                continue
            else:
                in_span = False
                filtered.append(msg)
                continue

        filtered.append(msg)

    return filtered
