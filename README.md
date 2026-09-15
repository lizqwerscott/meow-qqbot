# Meow QQBot

一个基于 QQ Botpy 框架的 QQ 机器人，集成了 OpenAI 接口的 AI 助手功能。

## 功能特性

- 支持私聊和群聊消息处理
- 集成 OpenAI GPT 模型，提供智能对话
- 消息队列管理，支持异步处理
- 聊天上下文管理，保持对话连贯性
- 支持流式响应和错误处理

## 项目结构

```
meow-qqbot/
├── main.py              # 主程序入口
├── config/               # 运行配置（不纳入版本控制）
│   ├── config.toml       # QQ 凭据与功能配置
│   ├── models.toml       # 模型 provider 与分组配置
│   └── allowlist.toml    # 权限与命令白名单
├── pyproject.toml       # 项目依赖配置
├── core/
│   ├── __init__.py
│   ├── client.py        # QQ 机器人客户端
│   ├── ai_service.py    # AI 服务模块
│   ├── context_manager.py  # 聊天上下文管理
│   └── message.py          # InputMessage 数据类
└── README.md
```

## 安装依赖

```bash
# 使用 uv 安装依赖
uv sync

# 或使用 pip
pip install -e .
```

## 配置

运行配置位于被 Git 忽略的 `config/`，请通过测试环境的密钥管理系统创建，不能提交
QQ `appid`、`secret` 或模型 API Key。至少准备：

```toml
# config/config.toml
appid = "测试 QQ Bot AppID"
secret = "测试 QQ Bot Secret"
character_card = "characters/default.md"
```

模型 provider 与分组配置放在 `config/models.toml`，权限规则放在
`config/allowlist.toml`。`character_card` 指向的角色卡也是运行资产，部署时必须
一并提供；缺失时服务会记录警告并以空角色卡继续运行。

测试服务器请使用独立 QQ Bot 凭据，并确保 `data/` 可写。首次启动会自动创建
身份和渠道缓存 SQLite 数据库；新测试环境无需复制生产身份库。

### S2-Pro 语音合成

启动 S2-Pro 服务后，在实际使用的 TOML 配置中添加：

```toml
[tts]
enabled = true
backend = "s2-pro"
base_url = "http://127.0.0.1:3030"
ref_audio = "characters/voice-reference.wav" # 可选，5-30 秒 WAV/MP3
ref_text = "参考音频对应的完整文本"           # 启用 ref_audio 克隆时必填；缺失则使用默认音色

[tts.s2_params]
max_new_tokens = 4096
temperature = 0.58
top_p = 0.88
top_k = 40
min_tokens_before_end = 0
```

`backend` 不配置时仍使用原有的 `voxcpm` 接口。S2-Pro 的情绪和效果标签必须直接写入
`text`，例如 `[excited] 今天真不错。`；它不支持 `instructions` 参数。正文不应包含半角/全角
分号或圆括号，服务会将分号替换为逗号并移除圆括号。服务返回的 32-bit float WAV 会自动归一化为
16-bit PCM WAV，以兼容 QQ 语音上传。`s2_params` 可选参数为 `max_new_tokens`、`temperature`、
`top_p`、`top_k` 和 `min_tokens_before_end`。工具定义会根据当前 `backend` 注入对应模型的正文、
情绪标签和 `voice_mode` 规则，避免将 VoxCPM 与 S2-Pro 的控制语法混用。

## 运行

```bash
uv run python main.py
```

## 使用说明

1. **私聊**：直接发送消息给机器人
2. **群聊**：在群聊中 @机器人 发送消息

机器人会自动处理消息，使用 OpenAI 模型生成回复，并保持对话上下文。

### 定时任务通知

定时任务成功后仅在产生最终可见结果时发送通知。`NO_REPLY`、`HEARTBEAT_OK` 和空结果表示静默完成，不会发送成功回执；任务执行状态与结果投递状态可分别通过任务详情查询。

## 核心模块说明

### 1. AI 服务模块 (`core/ai_service.py`)

- 使用 OpenAI 官方 Python 包
- 支持普通和流式响应
- 支持上下文管理
- 支持重试和错误处理

### 2. 消息数据类 (`core/message.py`)

- 异步消息队列管理
- 支持输入消息和处理后消息
- 自动清理旧消息

### 3. 上下文管理模块 (`core/context_manager.py`)

- 管理每个聊天的历史记录
- 支持消息数量限制
- 自动清理不活跃的聊天

### 4. 客户端模块 (`core/client.py`)

- 处理 QQ 机器人事件
- 集成 AI 服务
- 消息发送和接收

## 开发

### 添加新功能

1. 在 `core/ai_service.py` 中添加新的 AI 功能
2. 在 `core/client.py` 中处理相应的事件
3. 更新配置文件

### 调试

```python
# 设置日志级别
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 注意事项

1. **API 密钥安全**：不要将 API 密钥提交到版本控制系统
2. **速率限制**：注意 OpenAI API 的速率限制
3. **成本控制**：监控 API 使用情况，避免意外费用
4. **错误处理**：机器人有基本的错误处理，但建议添加监控

## 故障排除

### 常见问题

1. **无法连接 OpenAI API**
   - 检查 API 密钥是否正确
   - 检查网络连接
   - 检查 base_url 配置

2. **机器人不响应**
   - 检查 QQ 机器人配置
   - 检查日志输出
   - 确认机器人已上线

3. **上下文丢失**
   - 检查上下文管理器配置
   - 确认消息队列正常工作

## 许可证

MIT License
