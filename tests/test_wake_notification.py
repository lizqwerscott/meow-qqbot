"""run_wake_turn 通知决策回归测试。

背景：多轮工具循环的 wake 回合中，captured 会包含 AI 的中间话
（如“我去查一下”）和收尾报告。旧逻辑取第一条非静默回复，
导致只有中间话被投递、最终报告丢失。
修复：pick_final_notification_reply 取最后一轮非静默回复。
"""

import pytest

from core.engine.agent_engine import pick_final_notification_reply


def _silent(text: str) -> bool:
    return not text.strip() or text.strip().startswith(("好的", "收到", "明白了"))


def test_empty_captured_returns_empty():
    assert pick_final_notification_reply([], _silent) == ""


def test_all_silent_returns_empty():
    captured = ["好的", "收到", "明白了"]
    assert pick_final_notification_reply(captured, _silent) == ""


def test_single_reply_is_picked():
    captured = ["主人早安呀～"]
    assert pick_final_notification_reply(captured, _silent) == "主人早安呀～"


def test_last_non_silent_reply_is_picked():
    """回归场景：首轮是中间话，末轮才是真正报告。"""
    captured = [
        "主人～测试任务已经跑完啦！让猫猫看看执行结果喵！🔍",
        "测试结果出来啦主人！🎉\n\n**✅ 测试任务执行成功！**\n\n早安播报完美～",
    ]
    assert pick_final_notification_reply(captured, _silent) == captured[-1]


def test_trailing_silent_falls_back_to_previous_report():
    captured = [
        "我先查一下任务状态",
        "✅ 测试任务执行成功！",
        "好的",
    ]
    assert pick_final_notification_reply(captured, _silent) == "✅ 测试任务执行成功！"


def test_multi_round_tool_loop_picks_final_report():
    """多轮工具循环：中间话 + 数据整理 + 最终报告。"""
    captured = [
        "让猫猫看看执行结果喵！🔍",
        "数据已经拿到了，正在整理",
        "**最终报告**：测试任务执行成功，早安播报已发送（2382字）",
    ]
    assert pick_final_notification_reply(captured, _silent) == captured[-1]
