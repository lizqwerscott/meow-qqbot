import re
import sqlite3

import httpx
import pytest

from core.engine.engagement_config import EngagementConfig
from core.runtime_settings import (
    EngagementSettingsAdapter,
    InMemoryGroupTargetVerifier,
    RuntimeSettingsConflict,
    RuntimeSettingsCoordinator,
    RuntimeSettingsDegraded,
    RuntimeSettingsStore,
)
from core.webui.app import create_app


def _base_config():
    return {
        "mode_routing_enabled": True,
        "group_proactive_mode": "off",
        "group_proactive_active_chats": [],
    }


async def _coordinator(tmp_path, *, install=None):
    store = RuntimeSettingsStore(str(tmp_path / "runtime.sqlite3"))
    adapter = EngagementSettingsAdapter(
        _base_config(),
        store,
        capability_enabled=True,
        verifier=InMemoryGroupTargetVerifier({"g1"}),
        install=install,
    )
    coordinator = RuntimeSettingsCoordinator(store, adapter)
    await coordinator.initialize()
    return coordinator


@pytest.mark.asyncio
async def test_runtime_settings_requires_verified_target_for_active(tmp_path):
    coordinator = await _coordinator(tmp_path)
    snapshot = await coordinator.update(
        expected_revision=0,
        patch={
            "group_proactive_mode": "shadow",
            "group_proactive_active_chats": ["g1"],
        },
    )
    assert snapshot.revision == 1
    with pytest.raises(ValueError, match="unverified"):
        await coordinator.update(
            expected_revision=1,
            patch={"group_proactive_mode": "active"},
        )

    await coordinator.verify_target("g1")
    snapshot = await coordinator.update(
        expected_revision=1,
        patch={"group_proactive_mode": "active"},
    )
    assert snapshot.config.group_proactive_mode == "active"


@pytest.mark.asyncio
async def test_runtime_settings_cas_and_rollback(tmp_path):
    installed = []

    async def install(config: EngagementConfig):
        installed.append(config.group_proactive_interval_seconds)
        if config.group_proactive_interval_seconds == 70:
            raise RuntimeError("install failed")

    coordinator = await _coordinator(tmp_path, install=install)
    with pytest.raises(RuntimeError, match="install failed"):
        await coordinator.update(
            expected_revision=0,
            patch={"group_proactive_interval_seconds": 70},
        )
    assert coordinator.snapshot().revision == 0
    assert coordinator.snapshot().config.group_proactive_interval_seconds == 900

    await coordinator.update(
        expected_revision=0,
        patch={"group_proactive_interval_seconds": 80},
    )
    with pytest.raises(RuntimeSettingsConflict):
        await coordinator.update(
            expected_revision=0,
            patch={"group_proactive_interval_seconds": 40},
        )
    assert installed == [70, 900, 80]
    audits = await coordinator.audit(limit=10)
    assert [(item.outcome, item.failure_class) for item in audits[:2]] == [
        ("failure", "conflict"),
        ("success", None),
    ]
    assert any(item.failure_class == "install_failed" for item in audits)
    await coordinator.close()


@pytest.mark.asyncio
async def test_runtime_settings_persist_failure_restores_old_snapshot(tmp_path):
    coordinator = await _coordinator(tmp_path)
    store = coordinator._store

    async def fail_commit(*args, **kwargs):
        raise OSError("database unavailable")

    store.commit = fail_commit
    with pytest.raises(OSError, match="database unavailable"):
        await coordinator.update(
            expected_revision=0,
            patch={"group_proactive_interval_seconds": 80},
        )
    assert coordinator.snapshot().revision == 0
    assert coordinator.snapshot().config.group_proactive_interval_seconds == 900
    audits = await coordinator.audit(limit=10)
    assert any(item.failure_class == "persist_failed" for item in audits)
    await coordinator.close()


@pytest.mark.asyncio
async def test_runtime_settings_rollback_failure_enters_degraded_state(tmp_path):
    async def install(config: EngagementConfig):
        if config.group_proactive_interval_seconds in {70, 900}:
            raise RuntimeError("installation unavailable")

    coordinator = await _coordinator(tmp_path, install=install)
    with pytest.raises(RuntimeSettingsDegraded, match="rollback failed"):
        await coordinator.update(
            expected_revision=0,
            patch={"group_proactive_interval_seconds": 70},
        )
    assert coordinator.degraded
    assert coordinator.degraded_reason == "rollback_failed"
    with pytest.raises(RuntimeSettingsDegraded):
        await coordinator.update(
            expected_revision=0,
            patch={"group_proactive_interval_seconds": 80},
        )
    audits = await coordinator.audit(limit=10)
    assert {item.failure_class for item in audits} >= {
        "install_failed",
        "rollback_failed",
    }
    await coordinator.close()


@pytest.mark.asyncio
async def test_runtime_settings_target_remove_preserves_metadata(tmp_path):
    coordinator = await _coordinator(tmp_path)
    await coordinator.update(
        expected_revision=0,
        patch={"group_proactive_active_chats": ["g1"]},
    )
    target = await coordinator.remove_target("g1")
    assert target.verification_status == "removed"
    assert (await coordinator.targets())[0].verification_status == "removed"
    await coordinator.close()


@pytest.mark.asyncio
async def test_runtime_settings_migrates_v1_audit_schema(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE runtime_settings_schema (version INTEGER NOT NULL);
        INSERT INTO runtime_settings_schema(version) VALUES (1);
        CREATE TABLE runtime_settings_audit (
            id INTEGER PRIMARY KEY,
            domain TEXT NOT NULL,
            previous_revision INTEGER NOT NULL,
            new_revision INTEGER NOT NULL,
            change_json TEXT NOT NULL,
            source TEXT NOT NULL,
            remote_ip TEXT,
            created_at REAL NOT NULL
        );
        INSERT INTO runtime_settings_audit
            (domain, previous_revision, new_revision, change_json,
             source, remote_ip, created_at)
        VALUES ('engagement', 0, 1, '{"action":"update","fields":[]}',
                'webui', NULL, 1.0);
        CREATE TABLE runtime_settings (
            domain TEXT PRIMARY KEY,
            revision INTEGER NOT NULL,
            override_json TEXT NOT NULL,
            updated_at REAL NOT NULL,
            source TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'applied'
        );
        INSERT INTO runtime_settings
            (domain, revision, override_json, updated_at, source,
             schema_version, status)
        VALUES ('engagement', 1, '{}', 1.0, 'webui', 1, 'applied');
        """)
    connection.commit()
    connection.close()

    store = RuntimeSettingsStore(str(path))
    record = await store.get("engagement")
    audits = await store.list_audit("engagement")
    assert record.revision == 1
    assert record.schema_version == 2
    assert audits[0].outcome == "success"
    assert audits[0].failure_class is None
    await store.close()


@pytest.mark.asyncio
async def test_runtime_settings_audit_is_bounded_and_pageable(tmp_path):
    store = RuntimeSettingsStore(str(tmp_path / "runtime.sqlite3"), audit_retention=2)
    for index in range(3):
        await store.append_audit(
            "engagement",
            previous_revision=index,
            new_revision=index,
            change={"action": "test", "fields": [str(index)]},
            source="test",
            outcome="failure",
            failure_class="rejected",
        )
    first_page = await store.list_audit("engagement", limit=1)
    second_page = await store.list_audit(
        "engagement", limit=1, before_id=first_page[0].id
    )
    assert [item.change["fields"] for item in first_page + second_page] == [
        ["2"],
        ["1"],
    ]
    assert (
        await store.list_audit("engagement", limit=1, before_id=second_page[0].id) == []
    )
    await store.close()


@pytest.mark.asyncio
async def test_settings_webui_requires_token_and_csrf(tmp_path):
    coordinator = await _coordinator(tmp_path)
    app = create_app({"runtime_settings": coordinator}, {})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/settings/engagement/clear", data={"revision": 0})
    assert response.status_code == 403
    await coordinator.close()


@pytest.mark.asyncio
async def test_settings_webui_updates_with_token_and_csrf(tmp_path):
    coordinator = await _coordinator(tmp_path)
    app = create_app({"runtime_settings": coordinator}, {"token": "secret"})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/settings/engagement", headers={"Authorization": "Bearer secret"}
        )
        assert response.status_code == 200
        csrf = client.cookies.get("webui_csrf")
        assert csrf
        response = await client.post(
            "/settings/engagement/update",
            headers={"Authorization": "Bearer secret"},
            data={
                "_csrf": csrf,
                "revision": "0",
                "mode": "shadow",
                "active_chats": ["g1", "g2"],
                "interval_seconds": "900",
                "jitter_seconds": "0",
                "active_hours_start": "09:00",
                "active_hours_end": "23:00",
                "timezone": "Asia/Shanghai",
                "cooldown_seconds": "900",
                "quiet_cooldown_seconds": "300",
                "window_seconds": "3600",
                "max_turns_per_window": "2",
                "reservation_seconds": "120",
            },
        )
    assert response.status_code == 303
    assert coordinator.snapshot().config.group_proactive_mode == "shadow"
    assert coordinator.snapshot().config.group_proactive_active_chats == ("g1", "g2")
    await coordinator.close()


@pytest.mark.asyncio
async def test_settings_webui_group_picker_only_lists_group_chats(tmp_path):
    coordinator = await _coordinator(tmp_path)

    class ContextManager:
        async def get_all_disk_chat_ids_async(self):
            return ["group-1", "private-1", "group-2"]

        def get_chat_type(self, chat_id):
            return chat_id.startswith("group-")

    app = create_app(
        {"runtime_settings": coordinator, "context_manager": ContextManager()}, {}
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/settings/engagement")

    assert response.status_code == 200
    assert 'value="group-1"' in response.text
    assert 'value="group-2"' in response.text
    assert 'value="private-1"' not in response.text
    assert "搜索群 ID 或状态" in response.text
    await coordinator.close()


@pytest.mark.asyncio
async def test_settings_webui_active_mode_requires_server_confirmation(tmp_path):
    coordinator = await _coordinator(tmp_path)
    await coordinator.update(
        expected_revision=0,
        patch={
            "group_proactive_mode": "shadow",
            "group_proactive_active_chats": ["g1"],
        },
    )
    await coordinator.verify_target("g1")
    app = create_app({"runtime_settings": coordinator}, {"token": "secret"})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"Authorization": "Bearer secret"}
        page = await client.get("/settings/engagement", headers=headers)
        csrf = client.cookies.get("webui_csrf")
        active_nonce = re.search(
            r'name="active_nonce" value="([^"]+)"', page.text
        ).group(1)
        response = await client.post(
            "/settings/engagement/update",
            headers=headers,
            data={
                "_csrf": csrf,
                "revision": "1",
                "mode": "active",
                "active_chats": "g1",
                "interval_seconds": "900",
                "jitter_seconds": "0",
                "active_hours_start": "09:00",
                "active_hours_end": "23:00",
                "timezone": "Asia/Shanghai",
                "cooldown_seconds": "900",
                "quiet_cooldown_seconds": "300",
                "window_seconds": "3600",
                "max_turns_per_window": "2",
                "reservation_seconds": "120",
                "active_nonce": active_nonce,
            },
        )
        assert response.status_code == 200
        confirmation_nonce = re.search(
            r'name="confirmation_nonce" value="([^"]+)"', response.text
        ).group(1)
        response = await client.post(
            "/settings/engagement/confirm",
            headers=headers,
            data={
                "_csrf": csrf,
                "revision": "1",
                "confirmation_nonce": confirmation_nonce,
            },
        )
        assert response.status_code == 303
        assert coordinator.snapshot().config.group_proactive_mode == "active"
        response = await client.post(
            "/settings/engagement/pause",
            headers=headers,
            data={"_csrf": csrf, "revision": "2"},
        )
    assert response.status_code == 303
    assert coordinator.snapshot().config.group_proactive_mode == "off"
    await coordinator.close()
