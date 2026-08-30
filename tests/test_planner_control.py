import json

import pytest

from core.ai.deepseek_service import _convert_tools
from core.ai.protocol import AssistantMessage, AssistantToolCall
from core.engine.planner_control import (
    MAX_REASON_CHARS,
    PlannerAction,
    PlannerControlError,
    classify_planner_control_response,
    parse_planner_control,
    planner_control_tool,
)


def test_schema_is_a_strict_discriminated_union():
    schema = planner_control_tool()
    parameters = schema["function"]["parameters"]
    variants = parameters["oneOf"]

    assert schema["function"]["name"] == "planner_control"
    assert schema["function"]["strict"] is True
    assert parameters["type"] == "object"
    assert {variant["properties"]["action"]["const"] for variant in variants} == {
        "wait",
        "no_reply",
        "request_agent",
    }
    assert all(variant["additionalProperties"] is False for variant in variants)
    assert all(
        set(variant["required"]) == set(variant["properties"]) for variant in variants
    )


def test_deepseek_wire_schema_has_object_root():
    converted = _convert_tools([planner_control_tool()])

    assert converted is not None
    parameters = converted[0]["parameters"]
    assert parameters["type"] == "object"
    assert len(parameters["oneOf"]) == 3


def test_schema_can_exclude_agent_handoff_for_ambient_profiles():
    schema = planner_control_tool({PlannerAction.WAIT, PlannerAction.NO_REPLY})
    variants = schema["function"]["parameters"]["oneOf"]

    assert {variant["properties"]["action"]["const"] for variant in variants} == {
        "wait",
        "no_reply",
    }


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('{"action":"wait","wait_seconds":12,"reason":"waiting"}', PlannerAction.WAIT),
        ('{"action":"no_reply"}', PlannerAction.NO_REPLY),
        (
            '{"action":"request_agent","task_summary":"修复文件","reason":"需要工作区"}',
            PlannerAction.REQUEST_AGENT,
        ),
    ],
)
def test_parse_valid_planner_control_actions(arguments, expected):
    control = parse_planner_control(arguments)

    assert control.action is expected


def test_request_agent_requires_a_nonblank_summary_and_reason():
    with pytest.raises(PlannerControlError, match="task_summary cannot be blank"):
        parse_planner_control(
            {"action": "request_agent", "task_summary": " ", "reason": "需要执行"}
        )
    with pytest.raises(PlannerControlError, match="reason cannot be blank"):
        parse_planner_control(
            {"action": "request_agent", "task_summary": "修复文件", "reason": ""}
        )


@pytest.mark.parametrize(
    "arguments",
    [
        '{"action":"wait"}',
        '{"action":"wait","wait_seconds":0}',
        '{"action":"wait","wait_seconds":true}',
        '{"action":"no_reply","wait_seconds":1}',
        '{"action":"no_reply","unexpected":"value"}',
        '{"action":"no_reply","action":"wait","wait_seconds":1}',
        json.dumps({"action": "no_reply", "reason": "x" * (MAX_REASON_CHARS + 1)}),
    ],
)
def test_invalid_action_field_combinations_fail_closed(arguments):
    with pytest.raises(PlannerControlError):
        parse_planner_control(arguments)


def test_disallowed_action_fails_closed_for_ambient_profile():
    with pytest.raises(PlannerControlError, match="not allowed"):
        parse_planner_control(
            '{"action":"request_agent","task_summary":"修复","reason":"需要文件"}',
            allowed_actions={PlannerAction.WAIT, PlannerAction.NO_REPLY},
        )


def test_response_classifier_rejects_control_with_text_or_other_tool_call():
    with pytest.raises(PlannerControlError, match="visible assistant text"):
        classify_planner_control_response(
            AssistantMessage(
                content="我会处理",
                tool_calls=[
                    AssistantToolCall(
                        "call-1", "planner_control", '{"action":"no_reply"}'
                    )
                ],
            )
        )
    with pytest.raises(PlannerControlError, match="only tool call"):
        classify_planner_control_response(
            AssistantMessage(
                tool_calls=[
                    AssistantToolCall(
                        "call-1", "planner_control", '{"action":"no_reply"}'
                    ),
                    AssistantToolCall("call-2", "web_search", '{"query":"test"}'),
                ]
            )
        )


def test_response_classifier_leaves_normal_tool_calls_for_the_tool_loop():
    message = AssistantMessage(
        tool_calls=[AssistantToolCall("call-1", "web_search", '{"query":"test"}')]
    )

    assert classify_planner_control_response(message) is None
