"""Provider 工厂注册表测试（规格 §7 验收 2：注册路径自动化验证）。

覆盖：装饰器注册路径、ModelRegistry 集成（factory 被调用/服务入表/session_config）、
未知类型警告+跳过、factory 构造异常跳过（单模型失败不拖垮启动）。
"""

import pytest

from core.ai import provider_factory as pf
from core.ai.model_registry import ModelRegistry
from core.ai.provider_factory import get_provider_factory, register_provider


class _FakeService:
    """最小 LLMService 形状（协议测试用哨兵）。"""

    def __init__(self, model: str = "fake"):
        self.model = model
        self.base_url = None

    async def chat_completion(self, messages, **kwargs):
        return "hi", None

    async def chat_completion_with_tools(self, messages, **kwargs):
        return None, None

    async def close(self):
        pass


@pytest.fixture
def registered_factory():
    """注册一次性 mock provider，测试后必清理，避免污染其他测试。"""

    def factory(pcfg, mcfg):
        return _FakeService(model=mcfg.get("model", "fake"))

    pf._FACTORIES["mock-test"] = factory
    yield factory
    pf._FACTORIES.pop("mock-test", None)


def test_register_provider_decorator():
    """装饰器注册 → get_provider_factory 可查（注册路径本身）。"""

    @register_provider("mock-deco")
    def factory(pcfg, mcfg):
        return _FakeService()

    try:
        assert get_provider_factory("mock-deco") is factory
        assert "mock-deco" in pf._FACTORIES
    finally:
        pf._FACTORIES.pop("mock-deco", None)


def test_registry_uses_factory(registered_factory):
    """ModelRegistry 通过工厂构造服务（集成路径）。"""
    reg = ModelRegistry(
        {
            "mockp": {
                "type": "mock-test",
                "models": [{"name": "m1", "model": "fake-model"}],
            }
        },
        {},
    )
    svc = reg.get("mockp/m1")
    assert svc is not None
    assert isinstance(svc, _FakeService)
    assert svc.model == "fake-model"
    # 默认 session 配置（30 次 / 600s）也正确解析
    assert reg.get_session_config("mockp/m1") == (30, 600)


def test_unknown_provider_type_skipped(caplog):
    """未知 type → 警告 + 跳过（不再静默回退 openai）。"""
    reg = ModelRegistry({"bad": {"type": "nonexistent", "models": [{"name": "x"}]}}, {})
    assert len(reg._services) == 0
    assert "未知类型" in caplog.text


def test_factory_exception_skips_model(caplog, monkeypatch):
    """factory 抛异常 → 单模型跳过，不拖垮启动。"""

    def boom(pcfg, mcfg):
        raise RuntimeError("boom")

    monkeypatch.setitem(pf._FACTORIES, "mock-boom", boom)
    reg = ModelRegistry({"p": {"type": "mock-boom", "models": [{"name": "x"}]}}, {})
    assert len(reg._services) == 0
    assert "构造模型" in caplog.text
