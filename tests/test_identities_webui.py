import httpx
import pytest

from core.managers.identity_manager import IdentityManager
from core.session_identity import DeliveryTarget
from core.webui.app import create_app


@pytest.mark.asyncio
async def test_identity_page_focuses_the_requested_identity(tmp_path):
    identity_manager = IdentityManager(tmp_path / "identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "group-1")
    linked = identity_manager.observe(target, "openid-alpha", "小明")
    identity_manager.observe(target, "openid-beta", "小红")
    person_ref = identity_manager.create_person("大明")
    identity_manager.link_person(linked.identity_ref, person_ref, "test")

    app = create_app({"identity_manager": identity_manager}, {})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        full = await client.get("/identities")
        focused = await client.get(f"/identities?ref={linked.identity_ref}")
        missing = await client.get("/identities?ref=member_missing")

    assert full.status_code == 200
    assert "小明" in full.text
    assert "小红" in full.text
    assert f'id="identity-{linked.identity_ref}"' in full.text
    assert "row-focus" not in full.text
    assert "openid-alpha" not in full.text

    assert focused.status_code == 200
    assert "小明" in focused.text
    assert "小红" not in focused.text
    assert f'id="identity-{linked.identity_ref}"' in focused.text
    assert f'id="person-{person_ref}"' in focused.text
    assert "row-focus" in focused.text
    assert "显示全部身份" in focused.text
    assert "openid-alpha" not in focused.text

    assert missing.status_code == 200
    assert "身份库中未找到" in missing.text
    identity_manager.close()


@pytest.mark.asyncio
async def test_identity_page_lists_direct_peers_and_supports_focus(tmp_path):
    identity_manager = IdentityManager(tmp_path / "identity.sqlite3")
    group = DeliveryTarget("qq", "default", "group", "group-1")
    direct = DeliveryTarget("qq", "default", "direct", "peer-openid")
    group_member = identity_manager.observe(group, "openid-alpha", "小明")
    peer = identity_manager.observe(direct, "peer-openid", "老王")
    person_ref = identity_manager.create_person("某人")
    identity_manager.link_person(peer.identity_ref, person_ref, "test")

    app = create_app({"identity_manager": identity_manager}, {})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        full = await client.get("/identities")
        peer_focused = await client.get(f"/identities?ref={peer.identity_ref}")
        group_focused = await client.get(f"/identities?ref={group_member.identity_ref}")

    # 全量页：私聊区块展示对象，不泄露 openid
    assert full.status_code == 200
    assert "老王" in full.text
    assert f'id="identity-{peer.identity_ref}"' in full.text
    assert "peer-openid" not in full.text
    assert "openid-alpha" not in full.text

    # 私聊 focus：可定位，统一人物高亮，群成员被过滤
    assert peer_focused.status_code == 200
    assert "老王" in peer_focused.text
    assert "小明" not in peer_focused.text
    assert f'id="identity-{peer.identity_ref}"' in peer_focused.text
    assert f'id="person-{person_ref}"' in peer_focused.text
    assert "row-focus" in peer_focused.text
    assert "peer-openid" not in peer_focused.text

    # 群成员 focus：私聊区块不再展示私聊对象
    assert group_focused.status_code == 200
    assert "小明" in group_focused.text
    assert "老王" not in group_focused.text
    identity_manager.close()
