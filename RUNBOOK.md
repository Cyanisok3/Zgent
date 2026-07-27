# cyan 运维手册

## 用户入口

```bash
# 从项目根目录启动真实训练任务并进入 TUI
uv run cyan watch -- python train.py --config configs/base.yaml

# 重新附着最近的运行中或待处理任务
uv run cyan
```

`Ctrl+Q` 只 detach。只有 TUI 中显式按 `C` 才请求取消当前进程组。

## daemon

正常流程会自动启动 daemon。开发诊断时可单独操作：

```bash
uv run cyan core status
uv run cyan core start
uv run cyan core stop
uv run cyan ping
```

默认监听 `127.0.0.1:7437`，传输为 TCP loopback 上的 NDJSON/JSON-RPC 2.0；配置为
非 loopback 地址会在 daemon 启动前被拒绝。v1 不启动项目或全局配置中的 MCP server。

## 配置

优先级从低到高为：内建默认值、`~/.cyan/config.toml`、项目 `.cyan/config.toml`、`.env`、
系统环境变量。

```toml
[core]
host = "127.0.0.1"
port = 7437

[logging]
level = "INFO"
file = "~/.cyan/logs/core.log"
format = "text"
```

项目可选 smoke：

```toml
[incident.smoke]
argv = ["python", "smoke_train.py", "--steps", "2"]
timeout_s = 300
```

常用环境变量：

| 变量 | 默认值 |
|---|---|
| `CYAN_CONFIG` | `~/.cyan/config.toml` |
| `CYAN_HOST` | `127.0.0.1` |
| `CYAN_PORT` | `7437` |
| `CYAN_LOG_LEVEL` | `INFO` |
| `CYAN_LOG_FILE` | `~/.cyan/logs/core.log` |
| `CYAN_LOG_FORMAT` | `text` |
| `CYAN_LLM_DEFAULT_MODEL` | `claude-sonnet-4-6` |

## 诊断文件

```bash
tail -f ~/.cyan/logs/core.log
find ~/.cyan/jobs -maxdepth 4 -type f
```

- `launch.json` 为 `0600`，不得复制到 issue 或日志；
- `attempts/*/stdout.log` 与 `stderr.log` 是完整原始输出；
- `failure.json` 是失败时固定的 capsule；
- `incidents/*/proposal.diff` 只是 proposal，审批前不会修改 workspace。
- `incidents/*/smoke-execution.json` 保存可校验的 verifier PID 与启动身份；
- `traces/daemon.jsonl` 只记录日志、补丁、工具 I/O 和用户文本的字段/体积摘要。

## 常见状态

| 状态 | 含义 |
|---|---|
| `diagnosing` | 只读 Agent 正在调查 |
| `awaiting_approval` | diagnosis 与 diff 已就绪 |
| `smoke_running` | 正在运行用户声明的短验证 |
| `retry_running` | 已用原命令做最终验证 |
| `resolved` | 原命令真实退出码为 0 |
| `stale` | workspace hash 已变化，未应用补丁 |
| `rollback_blocked` | smoke 后文件又变化，自动回滚被安全阻止 |
| `unresolved` | 没有可信 proposal 或验证被中断 |

daemon 在 smoke 期间异常退出时，重启恢复会先用 PID、启动身份、session leader 和进程组
leader 四项校验遗留进程，确认终止后才把 Incident 置为 `unresolved` 并开放追问。

## 开发检查

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/unit -v
uv run pytest tests/integration -v
uv run python scripts/gen_protocol_doc.py --check
```
