import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.managers.workspace_manager import WorkspaceManager
from core.tools._types import ToolContext
from core.tools.deps import ToolDeps
from core.tools.impl.file import _approve_path_access


class FakePermissionManager:
    def __init__(self, role="admin"):
        self.role = role

    def get_user_role(self, sender_id):
        return self.role

    def is_admin_role(self, role):
        return role == "admin"


class FakeApprovalManager:
    def __init__(self, result="allow-once"):
        self.result = result
        self.request = None
        self.taken = []
        self.plan = None

    def check_whitelist(self, tool_name, target):
        return False

    async def request_approval(self, *args, **kwargs):
        self.request = (args, kwargs)
        self.plan = kwargs["plan"]
        return self.result, kwargs["session_key"]

    def take_pending_plan(self, session_key):
        self.taken.append(session_key)
        return self.plan


class RaisingApprovalManager(FakeApprovalManager):
    def __init__(self, error):
        super().__init__()
        self.error = error

    async def request_approval(self, *args, **kwargs):
        self.request = (args, kwargs)
        self.plan = kwargs["plan"]
        raise self.error


def _context(transition):
    return ToolContext(
        chat_id="admin-chat",
        is_group=False,
        reply_to="",
        sender_id="admin",
        reply_callback=lambda **kwargs: None,
        turn_id="turn-1",
        turn_revision=4,
        principal_id="admin",
        transition_turn=transition,
    )


def _deps(tmp_path, permission, approval):
    deps = ToolDeps(
        workspace_manager=WorkspaceManager(root=str(tmp_path / "workspaces")),
        permission_manager=permission,
    )
    deps.approval_manager.value = approval
    return deps


@pytest.mark.asyncio
async def test_path_approval_binds_turn_revision_and_resolved_path(tmp_path):
    permission = FakePermissionManager()
    approval = FakeApprovalManager()
    transitions = []

    async def transition(**kwargs):
        transitions.append(kwargs)
        return SimpleNamespace(revision=kwargs["expected_revision"] + 1)

    ctx = _context(transition)
    path = "../outside.txt"
    expected = Path(path).resolve()
    result = await _approve_path_access(
        ctx,
        path,
        "write_file",
        "outside sandbox",
        _deps(tmp_path, permission, approval),
    )

    assert result == expected
    assert [item["phase"].value for item in transitions] == [
        "awaiting_approval",
        "active",
    ]
    assert approval.request is not None
    assert approval.plan == {
        "tool_name": "write_file",
        "resolved_path": str(expected),
        "chat_id": "admin-chat",
        "turn_id": "turn-1",
        "principal_id": "admin",
        "turn_revision": 5,
        "approval_plan_id": approval.request[1]["session_key"],
    }
    assert ctx.turn_revision == 6
    assert approval.taken == [approval.request[1]["session_key"]]


@pytest.mark.asyncio
async def test_path_approval_rejects_when_turn_cannot_resume(tmp_path):
    permission = FakePermissionManager()
    approval = FakeApprovalManager()
    calls = 0

    async def transition(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            from core.engine.turn_state import TurnStateError

            raise TurnStateError("turn cancelled")
        return SimpleNamespace(revision=kwargs["expected_revision"] + 1)

    ctx = _context(transition)
    result = await _approve_path_access(
        ctx,
        "../outside.txt",
        "write_file",
        "outside sandbox",
        _deps(tmp_path, permission, approval),
    )

    assert result is None
    assert approval.taken == [approval.request[1]["session_key"]]


@pytest.mark.asyncio
async def test_path_approval_revalidates_admin_role_after_resolution(tmp_path):
    permission = FakePermissionManager()
    approval = FakeApprovalManager()

    async def transition(**kwargs):
        if kwargs["phase"].value == "active":
            permission.role = "default"
        return SimpleNamespace(revision=kwargs["expected_revision"] + 1)

    ctx = _context(transition)
    result = await _approve_path_access(
        ctx,
        "../outside.txt",
        "write_file",
        "outside sandbox",
        _deps(tmp_path, permission, approval),
    )

    assert result is None
    assert approval.taken == [approval.request[1]["session_key"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("approval transport failed")])
async def test_path_approval_restores_active_after_request_exception(tmp_path, error):
    permission = FakePermissionManager()
    approval = RaisingApprovalManager(error)
    transitions = []

    async def transition(**kwargs):
        transitions.append(kwargs)
        return SimpleNamespace(revision=kwargs["expected_revision"] + 1)

    with pytest.raises(RuntimeError, match="approval transport failed"):
        await _approve_path_access(
            _context(transition),
            "../outside.txt",
            "write_file",
            "outside sandbox",
            _deps(tmp_path, permission, approval),
        )

    assert [item["phase"].value for item in transitions] == [
        "awaiting_approval",
        "active",
    ]
    assert approval.taken == [approval.request[1]["session_key"]]


@pytest.mark.asyncio
async def test_path_approval_cancels_turn_when_request_is_cancelled(tmp_path):
    permission = FakePermissionManager()
    approval = RaisingApprovalManager(asyncio.CancelledError())
    transitions = []

    async def transition(**kwargs):
        transitions.append(kwargs)
        return SimpleNamespace(revision=kwargs["expected_revision"] + 1)

    with pytest.raises(asyncio.CancelledError):
        await _approve_path_access(
            _context(transition),
            "../outside.txt",
            "write_file",
            "outside sandbox",
            _deps(tmp_path, permission, approval),
        )

    assert [item["phase"].value for item in transitions] == [
        "awaiting_approval",
        "cancelled",
    ]
    assert approval.taken == [approval.request[1]["session_key"]]
