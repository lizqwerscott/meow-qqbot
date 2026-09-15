import httpx
import pytest

from core.managers.identity_manager import IdentityManager
from core.session_identity import DeliveryTarget
from core.session_identity.channel_info import QQChannelInfoProvider
from core.webui.app import create_app


class _FakeApi:
    async def request(self, method, path, timeout=0):
        if path == "/users/@me":
            return {"id": "bot", "display_name": "猫猫"}
        if path.endswith("/info"):
            return {
                "group_openid": "group-1",
                "group_name": "测试群",
                "description": "群聊简介",
                "member_count": 8,
            }
        return {"member_openid": "bot", "role": "admin", "can_send": True}


@pytest.mark.asyncio
async def test_channels_page_renders_cached_group_information(tmp_path):
    identity_manager = IdentityManager(tmp_path / "identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "group-1")
    identity_manager.observe(target, "member-1", "测试成员", observed_at=1_700_000_000)

    channel_info_provider = QQChannelInfoProvider(
        _FakeApi(), db_path=tmp_path / "channel_info.sqlite3"
    )
    await channel_info_provider.get_info(target)

    app = create_app(
        {
            "identity_manager": identity_manager,
            "channel_info_provider": channel_info_provider,
        },
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/channels")

    assert response.status_code == 200
    assert "测试群" in response.text
    assert "群聊简介" in response.text
    assert "8" in response.text
    assert "测试成员" not in response.text
    assert "group-1" not in response.text
