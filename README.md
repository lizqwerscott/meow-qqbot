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
├── config.toml          # 配置文件
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

1. 复制配置文件模板：
```bash
cp config.toml.example config.toml
```

2. 编辑 `config.toml` 文件：
```yaml
appid: "你的QQ机器人AppID"
secret: "你的QQ机器人Secret"

# OpenAI 配置
openai:
  api_key: "你的OpenAI API Key"  # 必填
  base_url: "https://api.openai.com/v1"  # 可替换为其他兼容接口
  model: "gpt-3.5-turbo"  # 模型名称
  temperature: 0.7  # 温度参数
  max_tokens: 1000  # 最大生成token数
  timeout: 30  # 请求超时时间
  max_retries: 3  # 最大重试次数

# AI 系统提示
ai_system_prompt: |
  你是一个友好的QQ机器人助手，请用中文回答用户的问题。
  保持回答简洁、有帮助，避免冗长。
```

3. 设置环境变量（可选）：
```bash
export OPENAI_API_KEY="你的OpenAI API Key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-3.5-turbo"
```

## 运行

```bash
python main.py
```

## 使用说明

1. **私聊**：直接发送消息给机器人
2. **群聊**：在群聊中 @机器人 发送消息

机器人会自动处理消息，使用 OpenAI 模型生成回复，并保持对话上下文。

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
