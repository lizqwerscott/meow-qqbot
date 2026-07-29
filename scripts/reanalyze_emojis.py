"""重新分析所有 VLM 分析失败的表情。

读取 config.toml 中的多模态配置（支持 group fallback 链），
逐个调用 reanalyze_emoji。

用法:
    uv run python scripts/reanalyze_emojis.py
    uv run python scripts/reanalyze_emojis.py --force    # 强制重新分析全部
"""

import argparse
import asyncio
import logging
import sys
import tomllib
from pathlib import Path

# 确保项目根目录在 sys.path 中，使 core 模块可导入
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import httpx

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(message)s",
)
_log = logging.getLogger("reanalyze")


async def main():
    parser = argparse.ArgumentParser(description="重新分析所有分析失败的表情")
    parser.add_argument("--force", action="store_true", help="强制重新分析所有表情（含已成功的）")
    parser.add_argument("--config", default="config/config.toml", help="配置文件路径")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    from core.ai.model_registry import ModelRegistry
    from core.ai.multimodal import MultimodalService
    from core.managers.emoji_manager import EmojiManager

    providers_config = raw.get("providers", {})
    groups_config = raw.get("groups", {})
    multimodal_cfg = raw.get("multimodal", {})

    services = []
    model_names = []

    def _add_service(svc, name):
        services.append(svc)
        model_names.append(name)

    # ── 策略 1: 新格式 [providers] + [groups] ─────────────────
    if providers_config and groups_config:
        registry = ModelRegistry(providers_config, groups_config)
        multimodal_group = multimodal_cfg.get("group", "multimodal")
        chain = registry.get_chain(multimodal_group)

        if chain:
            for qualified_name in chain:
                svc = registry.get(qualified_name)
                if svc is not None:
                    _add_service(svc, qualified_name)
            print("多模态模型链（按 fallback 顺序）:")
            for qn in chain:
                s = registry.get(qn)
                ok = "✓" if s is not None else "✗ 未注册"
                print(f"  {ok}  {qn}")

    # ── 策略 2: 旧格式 [models.xxx] 中的 ollama 模型 ──────────
    if not services and raw.get("models"):
        from core.ai.service import AIService

        for mname, mcfg in raw["models"].items():
            if mcfg.get("provider") == "ollama":
                model = mcfg.get("model", "")
                if not model:
                    continue
                host = mcfg.get("host", "http://localhost:11434").rstrip("/")
                svc = AIService(
                    api_key=mcfg.get("api_key", "") or "not-needed",
                    base_url=f"{host}/v1",
                    model=model,
                    timeout=mcfg.get("timeout", 120),
                    max_retries=0,
                    temperature=mcfg.get("temperature", 0.3),
                    max_tokens=mcfg.get("max_tokens", 4096),
                )
                _add_service(svc, f"models.{mname}")
                print(f"  ✓  [models.{mname}]  ollama: {model}")

    # ── 策略 3: [multimodal] 直接配置 ─────────────────────────
    if not services:
        model = multimodal_cfg.get("model", "")
        if model:
            from core.ai.service import AIService
            host = multimodal_cfg.get("host", "http://localhost:11434").rstrip("/")
            svc = AIService(
                api_key=multimodal_cfg.get("api_key", "") or "not-needed",
                base_url=f"{host}/v1",
                model=model,
                timeout=120,
                max_retries=0,
                temperature=0.3,
                max_tokens=4096,
            )
            _add_service(svc, model)
            print(f"  ✓  {model} @ {host}")

    if not services:
        print("错误: 无法确定 VLM 模型。请配置以下任一:")
        print("  1. [providers] + [groups]（新格式，推荐）")
        print("  2. [models.xxx] provider='ollama'（旧格式）")
        print("  3. [multimodal] model=... host=...（直接配置）")
        sys.exit(1)

    mm_svc = MultimodalService(services, model_names=model_names)

    http_client = httpx.AsyncClient(timeout=60.0)
    manager = EmojiManager(
        http_client=http_client,
        multimodal_service=mm_svc,
    )

    all_emojis = manager._storage.list_all()
    if not all_emojis:
        print("\n表情数据库为空，无需操作。")
        return

    if args.force:
        pending = all_emojis
        reason = f"强制重新分析全部（共 {len(pending)} 个）"
    else:
        pending = [e for e in all_emojis if not e.get("auto_summary")]
        reason = f"auto_summary 为空（共 {len(pending)} 个）"

    if not pending:
        print(f"\n所有 {len(all_emojis)} 个表情均已分析完成，无需重新分析。")
        return

    print(f"\n共 {len(all_emojis)} 个表情，{reason}")
    print()

    success = 0
    failed = 0
    for i, emoji in enumerate(pending, 1):
        h = emoji["hash"]
        short = h[:12]
        file_name = emoji.get("file_name", "?")

        print(f"  [{i:>3}/{len(pending)}] {short}  {file_name:<30}", end=" ", flush=True)
        ok = await manager.reanalyze_emoji(h)
        if ok:
            success += 1
            print("✓")
        else:
            failed += 1
            print("✗")

    print()
    print(f"完成: {success} 成功, {failed} 失败 / {len(pending)} 个")

    await http_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
