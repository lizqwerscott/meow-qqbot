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
- `core/ai/service.py` — OpenAI-compatible LLM client (`openai[aiohttp]`)
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

- **Per-session isolation**: `SessionTaskManager` creates separate `asyncio.Queue` + `asyncio.Lock` per `chat_id`. Messages within the same session are processed serially.
- **Message dedup**: `AgentEngine._processed_ids` (OrderedDict, LRU cap 1000) prevents WS reconnect double-processing.
- **Tool loop**: `ToolLoop` runs up to N rounds of AI → tool_calls → execute → feed back (configurable via `max_tool_rounds`, default unlimited = -1). Every text response is sent immediately via `reply_callback`.
- **Auto memory injection**: `_build_memory_context` fetches relevant Hindsight episodes/profiles and injects into the dynamic system prompt (invisible to user). Up to 3 episodes (150 chars each) + 1 profile; dirty data filtered.
- **Prompt structure**: Static prompt (role + character card + skill intro + memory desc + tool coop + emoji guide) rendered once from Jinja2 templates (`prompts/`). Dynamic block (skill entries + memory context + learning context + time + emoji tags + users list + workspace info) appended as a separate system message each turn.
- **Multi-model routing** (optional): `RuleRouter` scores messages on 15 dimensions (code, complexity, reasoning, etc.) and assigns SIMPLE / MEDIUM / COMPLEX / REASONING tiers. Each tier has a fallback chain of models from `[models]` config. Routing disabled by default (`[routing].enabled = false`).
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
