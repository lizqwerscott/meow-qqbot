from types import SimpleNamespace

import pytest

from core.engine.chat_type_index import ChatTypeIndex
from core.session_identity import DeliveryTarget, DeliveryTargetCatalog


def _group(chat_id: str) -> DeliveryTarget:
    return DeliveryTarget("qq", "default", "group", chat_id)


def _direct(chat_id: str) -> DeliveryTarget:
    return DeliveryTarget("qq", "default", "direct", chat_id)


def test_catalog_get_chat_type_returns_kind():
    catalog = DeliveryTargetCatalog(":memory:")
    catalog.observe(_group("g1"))
    catalog.observe(_direct("u1"))

    assert catalog.get_chat_type("g1") == "group"
    assert catalog.get_chat_type("u1") == "direct"
    assert catalog.get_chat_type("missing") is None
    assert catalog.get_chat_type("") is None


def test_catalog_get_chat_type_is_fail_safe_on_ambiguity():
    catalog = DeliveryTargetCatalog(":memory:")
    catalog.observe(_group("x"))
    catalog.observe(_direct("x"))

    assert catalog.get_chat_type("x") is None


def test_index_get_uses_catalog_for_raw_target_id():
    catalog = DeliveryTargetCatalog(":memory:")
    index = ChatTypeIndex(catalog)
    index.observe(_group("g1"))
    index.observe(_direct("u1"))

    assert index.get("g1") is True
    assert index.get("u1") is False
    assert index.get("missing") is None


def test_index_observe_is_idempotent():
    catalog = DeliveryTargetCatalog(":memory:")
    index = ChatTypeIndex(catalog)

    index.observe(_group("g1"))
    index.observe(_group("g1"))

    assert index.get("g1") is True


def test_index_observe_ignores_missing_catalog_and_target():
    index = ChatTypeIndex()
    index.observe(_group("g1"))
    index.observe(None)
    assert index.get("g1") is None


def test_index_get_falls_back_to_resolver_for_canonical_key():
    catalog = DeliveryTargetCatalog(":memory:")

    class Resolver:
        def resolve_legacy(self, key):
            if key == "agent:main:qq:group:g1":
                return SimpleNamespace(target=SimpleNamespace(chat_type="group"))
            if key == "agent:main:qq:direct:u1":
                return SimpleNamespace(target=SimpleNamespace(chat_type="direct"))
            raise ValueError("unknown legacy session key")

    index = ChatTypeIndex(catalog, Resolver())

    assert index.get("agent:main:qq:group:g1") is True
    assert index.get("agent:main:qq:direct:u1") is False
    assert index.get("agent:main:qq:group:unknown") is None


def test_index_get_returns_none_for_internal_session():
    catalog = DeliveryTargetCatalog(":memory:")

    class Resolver:
        def resolve_legacy(self, key):
            return SimpleNamespace(target=None)

    index = ChatTypeIndex(catalog, Resolver())
    assert index.get("heartbeat:events") is None
