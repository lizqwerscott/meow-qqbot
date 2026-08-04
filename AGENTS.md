# meow-qqbot — AGENTS.md

## Run & Dev

```bash
uv sync               # install deps (Python 3.11+)
uv run python main.py # run the bot
uv add <package>      # add a dependency
uv run <script.py>    # run any script in the venv

## Formatting

```bash
uv run isort <file>   # sort imports
uv run black <file>   # format code
```
```

## Entrypoint & Architecture

- `main.py` — bootstrap entrypoint; wires services in cascade, then starts WebSocket
- `config.toml` — **contains real secrets** (`appid`, `secret`, API keys). Do not commit or expose.
- `core/engine/client.py` — `BotEngine`: WebSocket lifecycle, message I/O via `qqbot-agent-sdk`
- `core/engine/agent_engine.py` — `AgentEngine` (global singleton): AI orchestration, tool loop, session queues, prompt assembly
- `core/engine/router.py` — `Router`: distinguishes commands from AI dialogue, dispatches to AgentEngine
- `core/engine/prompt_builder.py` — Prompt assembly (Jinja2 templates in `prompts/`)
- `core/engine/hindsight_memory.py` — `HindsightMemory`: async HTTP client for Hindsight (`http://127.0.0.1:8888`)
- `core/engine/duplicate_reply.py` — Duplicate message detection/echoback in groups
- `core/ai/service.py` — OpenAI-compatible LLM client (`openai[aiohttp]`); returns unified `AssistantMessage` protocol objects
- `core/ai/protocol.py` — **AI 协议抽象层**: `AssistantMessage` / `AssistantToolCall` 统一消息对象（`tool_calls_data` wire 组装）+ `ensure_messages_consistent` 一致性清理。核心循环（ToolLoop/FallbackRunner）只依赖本模块，不感知底层协议
- `core/ai/provider_factory.py` — **Provider 工厂注册表**: `@register_provider(type)` 自注册构造器，`ModelRegistry` 构造零分支。新增 provider = 写 factory + 配置，不动注册表
- `core/ai/multimodal.py` — VLM vision model client for emoji/image analysis
- `core/ai/model_registry.py` — `ModelRegistry`: multi-model config + fallback chains
- `core/managers/` — `*Manager` classes (command, context, cost, emoji, nickname, session, template, permission, workspace)
- `core/tools/` — tool definitions, `ToolExecutor`, skill managers, `ToolLoop`
- `core/plugins/` — plugin system (`PluginManager`, `BasePlugin`)
- `core/learners/` — learning system (`LearningOrchestrator` + `JargonMiner`, `BehaviorLearner`, `ExpressionLearner`, `SceneClusterer`)
- `core/rule_router.py` — `RuleRouter`: 15-dimension scoring engine (ClawRouter-style) for model-tier routing
- `core/router_model.py` — Lightweight router model (qwen2.5:7b) for smart tier routing: simple tasks reply directly, complex tasks mark `[ESCALATE]`
- `core/message.py` — Unified message model (`MessageType`, `ResourceMeta`) for all resource types (emoji/image/voice/video/file)
- `core/card_parser.py` — QQ card message parser (ARK/EMBED → unified share text format)
- `core/image_utils.py` — Image preprocessing utilities (normalize, resize, format conversion for VLM consumption)
- `core/web_search/` — Web search/fetch service: `config.py` (config parse), `providers.py` (Ollama/Tavily/DuckDuckGo impls), `service.py` (`WebService`: provider chain + fallback + cache + SSRF)
- `core/approval/` — `ApprovalManager`: operation approval system with file-path/exec-command whitelist, persisted to `config/approval_whitelist.json`
- `core/tasks/` — background task system (`TaskManager`, `CronJobManager`, `CronJobScheduler`)
- `core/webui/` — FastAPI + Jinja2 management panel (optional, enabled via config)
- `allowlist.toml` — role-based permissions + command whitelist

## Key Dependencies

- `qqbot-agent-sdk` — from private git repo `github.com/lizqwerscott/qqbot-agent-sdk`, branch `group-message-add` (pinned in `pyproject.toml` via `[tool.uv.sources]`)
- **Hindsight** (`hindsight-client>=0.8.4`) — external long-term memory at `http://127.0.0.1:8888`; health-checked at startup. Degrades gracefully if unreachable.
- `openai[aiohttp]` — >= 2.26.0
- `skillkit>=0.4.0` — skill execution runtime

## Architecture Notes

- **Web search/fetch** (optional, `[web_search]`/`[web_fetch]` config sections): two tools — `web_search` (query/count/region/freshness, 1-10 results) and `web_fetch` (URL → title/content/links, SSRF-guarded). Providers: Ollama (`OLLAMA_API_KEY`/`OLLAMA_BASE_URL`; local host免 key, fallback链 experimental→local→hosted), Tavily (`TAVILY_API_KEY`), DuckDuckGo (key-free, HTML parse). `providers` list is the explicit fallback chain (single entry = locked provider); `strict_credential_skip=true` skips uncredentialed providers; `fallback_on_empty=false` means only failures (not empty results) trigger fallback. 15-min in-memory cache, results tagged with `provider`. `web_fetch` chain is local-first (`local` → ollama → tavily). `block_private_ip` SSRF guard with `allow_fake_ip_range` (default true) for Clash/Surge fake-ip DNS (`198.18.0.0/15`, `fc00::/7`).
- **Per-session isolation**: `SessionTaskManager` creates separate `asyncio.Queue` + `asyncio.Lock` per `chat_id`. Messages within the same session are processed serially.
- **Message dedup**: `AgentEngine._processed_ids` (OrderedDict, LRU cap 1000) prevents WS reconnect double-processing.
- **Tool loop**: `ToolLoop` runs up to N rounds of AI → tool_calls → execute → feed back (configurable via `max_tool_rounds`, default unlimited = -1). Every text response is sent immediately via `reply_callback`.
- **Auto memory injection**: `_build_memory_context` fetches relevant Hindsight episodes/profiles and injects into the dynamic system prompt (invisible to user). Up to 3 episodes (150 chars each) + 1 profile; dirty data filtered.
- **Prompt structure**: Static prompt (role + character card + skill intro + memory desc + tool coop + emoji guide) rendered once from Jinja2 templates (`prompts/`). Dynamic block (skill entries + memory context + learning context + time + emoji tags + users list + workspace info) appended as a separate system message each turn.
- **Multi-model routing** (optional): `RuleRouter` scores messages on 16 dimensions (code, complexity, reasoning, etc.) and assigns SIMPLE / MEDIUM / COMPLEX / REASONING tiers. Each tier has a fallback chain of models from `[models]` config. Routing disabled by default (`[routing].enabled = false`).
- **Keyword flush**: Certain keywords (`"我喜欢"`, `"记住"`, etc.) trigger an immediate Hindsight `flush` for the current session.

## Bot Commands

- All commands are prefixed with `猫猫` (e.g., `猫猫状态`, `猫猫表情列表`)
- In groups: messages starting with `猫猫` trigger AI reply even without `@mention`
- Command handlers in `core/command_handlers/`: `status`, `help`, `history`, `skills`, `tts`, `cost`, `tasks`, `cron`, `archive`, `jargon`, `plugin_mgmt`, `heartbeat`, `emoji_info`, `emoji_list`, `emoji_edit`, `emoji_reset`, `approval_test`, `base`
- Admin-only commands: `猫猫状态`, `猫猫表情编辑`, `猫猫表情重置`
- Nicknames stored in `nicknames.json` (manual) and `data/nicknames.json` (auto-collected, debounced 10s save)

## Hindsight (Long-term Memory)

- External HTTP service at `http://127.0.0.1:8888` (not the EverOS service documented in `api.md`).
- Single bank (`bank_id: "qq_bot"`). All messages for a session share a `document_id` (`session-{chat_id}`) in append mode.
- User isolation via tags: `user:{sender_id}`. Recall uses `tags_match=all_strict`.
- `retain_async=True` — all `add_message` calls are fire-and-forget.
- Config: `hindsight.base_url`, `hindsight.bank_id`, `hindsight.search_top_k`.
- At startup `main.py` performs a health check; if it fails, memory features degrade gracefully.

## Permissions

- Configured in `allowlist.toml`.
- Roles (hierarchical): `system` > `admin` > `trusted` > `default`.
- Each tool has a minimum role requirement (`all` / `trusted` / `admin` / `system`).
- `execute_command` allows all users but the command itself is validated against the whitelist (`[commands].allowed`).
- Security policies: `deny_chaining` (no `;` `&&` `||` for non-admin), `deny_redirect`, `max_command_length`, etc.

## Exec 审批策略（OpenClaw 风格）

- 三层模型：`core/approval/exec_policy.py`（策略面 + `requires_approval` 判定）→ `core/tools/exec_analysis.py`（shell 链切段 + 真实路径解析 + inline-eval 检测）→ `core/approval/allowlist.py`（路径 glob + argPattern 匹配）。
- 策略面：`[exec]` 段（config/allowlist.toml）为 requested policy，`config/approval_whitelist.json` 的 `defaults` 为 host policy，`effective_policy` 取更严。`mode`: `deny | allowlist | ask | auto | full`。
- 角色归一（`policy_for_role`）：`system` → full；`trusted/default` → allowlist+off（miss 直接拒，不弹卡）；`admin` → 用 `[exec]` 配置（默认 on-miss，可审批）。
- 命令分析：**tree-sitter-bash CST 切段**（`core/tools/bash_cst.py`），按 `&& || ; | &` 切段（尾随重定向不塌缩链），每段独立 PATH 解析 + allowlist 匹配（bare name 只匹配 PATH 解析结果，路径 glob 支持 `**`/`~`，`arg_pattern` 正则约束参数）；`$(...)`/反引号/`<(...)`/`bash -c` payload 内部命令递归分析（深度 2），内部命令也要命中 allowlist；语法错误 fail-closed 拒绝。**无命令黑名单**（对齐 OpenClaw）：危险命令由 allowlist 覆盖率 + 审批 + auto-review 承担，不在准入层硬编码命令名。
- **heredoc 检测**（对齐 openclaw `reason: "heredoc"`）：段内含 `<<EOF`（CST `heredoc_redirect` 节点）即使 allowlist 命中也要走审批——heredoc 可嵌入任意多行脚本内容，且 shell=False 下本就不生效（token 当参数）。
- **与 openclaw 的剩余差距与修改方案**：见 `docs/exec-review-gap.md`（包装器解包 / 解释器绑定精度 / 审批文本兜底+转发 / 审批管理命令 / 全量 shell 语义远期）。
- **safe bins**（`[exec]` 段 `safe_bins`，对齐 openclaw `tools.exec.safeBins`）：预信任窄 stdin 过滤器（内置默认 profiles：head/tail/wc/tr；`safe_bin_profiles` 可覆盖），命中且 argv 满足 profile（`max_positional`/`allowed_value_flags`/`allowed_flags`/`denied_flags`）的段视为 allowlist 满足，管道场景（`ls | head -5`）无需白名单条目。
- **审批超时 followup**：`[exec]` 段 `approval_timeout`（默认 300s，对齐 openclaw pending 过期）；后台执行审批超时/拒绝时向 delivery_channel 投递 followup 通知（对齐 openclaw "命令未运行" 会话恢复）。
- `strict_inline_eval`: `python -c` / `node -e` / `osascript -e` 等内联求值即使二进制在白名单也强制审批，且 allow-always 不落白名单（`persist=False`）。
- **exec `env` 参数**（对齐 OpenClaw）：模型可直接给 exec 传 `env`（`{KEY: value}`）注入子进程环境，避免包 `bash -c 'export K=V && ...'`（后者触发 strictInlineEval 门禁）。危险键/PATH/非法键名硬拒绝（`Security Violation`）；env 覆盖子集进入 plan 绑定（审批时比对防漂移，只绑模型传入的覆盖、不绑非确定的 login-shell 基础环境）。危险键/前缀表集中维护在 `core/tools/env_override_policy.py`。此表是**有意添加的 env 覆盖黑名单**——与命令层"无黑名单"哲学不同：env 注入无法靠命令 allowlist/审批兜底（环境变量不触发命令审批，PATH/LD_*/PYTHONPATH 可劫持二进制解析），因此这里是命令行黑名单原则之外的**正当安全例外**。
- allowlist 命中直跑；miss 时 admin 私聊弹审批卡（`ask_fallback` 默认 deny），其余拒绝。前台群聊不弹卡（审批/审查仅限 c2c）；**后台执行（`background=true`）例外**——对齐 OpenClaw：interactive chat 中的 background exec 走同一审批流，审批卡投递 admin c2c，通过后才 spawn，避免"审批不到直接失败"；auto-reviewer 是模型判定不依赖聊天面，同样放行。
- 非 admin 仍叠加 `PermissionManager.check_command_allowed`（替换/串联/管道/重定向/长度/`[commands].allowed`）。
- `mode=auto` 可接 `ExecAutoReviewer`（`core/approval/auto_reviewer.py`，通过 `deps.exec_reviewer` 注入），miss 先 LLM 审查再转人工。
- **段级执行**（`core/tools/exec_runner.py`）：前台执行按分析结果逐段跑（shell=False），`&&`/`||` 短路、`;` 顺序、`|` 管道（PIPE 级联），argv[0] 用解析后真实路径（pin executable）；链式命令不再把 `&&` 当参数；后台模式不支持链式（明确报错）。
- **durable plan 绑定**：审批时 plan（command/cwd/resolved_path）存入 `ApprovalManager._pending_plans`，审批通过后执行前经 `take_pending_plan` 比对，不一致返回 `APPROVAL_MISMATCH`（对齐 openclaw approval mismatch）。
- `config/approval_whitelist.json` 为 v2 schema（`version`/`defaults`/`allowlist`/`file_paths`），v1 `exec_commands` 自动迁移（source=legacy），旧字段保留镜像兼容。

## Background Tasks

- Optional task system (`[tasks]` config section). Persists to `data/tasks/`.
- One-shot tasks (`TaskManager`) and cron jobs (`CronJobManager`, `CronJobScheduler`).
- AI can create/cancel tasks via tools (`cron`, `task`).
- Cron scheduler runs in background; triggers AI execution with task-specific prompt (`task_chat.j2`).

## Heartbeat

- Optional (`[heartbeat]` config section). Runs periodic AI-driven health checks.
- Configurable schedule (`every` in seconds), model fallback chain, active hours.
- Admin's HEARTBEAT.md in `workspaces/` defines the inspection checklist.
- Two system-prompt modes: `minimal` (dedicated simple prompt) or `normal` (reuses character card).

## Learning System

- Optional (`[learners]` config section). Persists to `data/learners/`.
- Components: `JargonMiner` (extracts group-specific slang/terms), `BehaviorLearner`, `ExpressionLearner`, `SceneClusterer`.
- Learner context auto-injected into dynamic prompt on each turn (if data exists).

## Plugins

- `PluginManager` loads Python plugins from `plugins/` directory.
- `BasePlugin` interface with `on_load` / `on_unload` hooks.
- Current installed: `pighub/`.

## WebUI Management Panel

Built-in FastAPI + Jinja2 web UI at `core/webui/`. Enabled via config:

```toml
[webui]
enabled = true
host = "0.0.0.0"
port = 8090
token = "your-secret-token"
```

Access at `http://<host>:8090`. Features: system status, emoji CRUD, nickname management, session history browse, learner data viewer, cost tracking. Embeds in the main process as a background uvicorn task.

## Workspace & File Tools

- `WorkspaceManager` (`core/managers/workspace_manager.py`) — manages per-chat sandboxed directories under `workspaces/`.
- Each chat (group or private) gets its own `workspaces/{groups,private}/{chat_id}/files/` sandbox.
- Four file tools (`read_file`, `write_file`, `edit_file`, `apply_patch`) are sandboxed per-chat — path traversal is blocked. Search file content with `execute_command + rg`.
- `workspaces/HEARTBEAT.md` — global heartbeat prompt file, read by `HeartbeatManager` at runtime.
- Admin's private chat gets workspace-wide file access (entire `workspaces/` root).
- `execute_command` remains unrestricted for admin; file tool sandbox applies to all users.

## Known Gotchas

- `config.toml` has real credentials (`appid`, `secret`, API keys) — never commit or expose it. No `.example` counterpart exists.
- `api.md` in the repo root is the **EverOS API spec** (not this bot's API). The bot uses **Hindsight** (`hindsight-client`), not EverOS.
- Template paths are hardcoded in `TemplateManager`. Character card path configured via `character_card: characters/default.md`.
- Cache dirs in `.gitignore`: `cache/*`, `data/*` (except `.gitkeep`), `workspaces/*`.
- `.agents/skills/` is the project skill directory (loaded by `SkillManagers` at startup).
- `config/` is gitignored — approval whitelist at `config/approval_whitelist.json` is auto-created at runtime, not tracked.
- `skills-lock.json` is auto-generated and gitignored (regenerated via `skill install`/`sync`).
