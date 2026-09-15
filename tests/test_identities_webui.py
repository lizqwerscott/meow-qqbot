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
