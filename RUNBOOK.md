# cyan 运维手册

## 用户流程

从模型项目根目录运行 `cyan`，然后：

```text
/monitor
# 粘贴训练命令并检查预览
/start
```

本地命令为 `/monitor`、`/start`、`/jobs`、`/incident <text>` 和 `/help`。补丁批准、拒绝
和训练取消不是 slash command，只在对应状态出现：

- 运行中：`Cancel training process`
- 有 smoke：`Approve and run smoke`、`Approve without smoke`、`Reject patch`
- 无 smoke：`Approve and rerun original command`、`Reject patch`
- 非 Git workspace：只允许 `Reject patch`

选择器支持鼠标或上下键和 Enter。`Esc` 返回输入框，`Ctrl+Q` 只分离 TUI。

## daemon 与配置

正常流程自动启动 daemon。daemon 生命周期命令：

```bash
cyan core start [--json]
cyan core stop
```

内部 daemon 入口是 `cyan-core`；`cyan` 会自动确保 daemon 启动后打开 TUI。daemon 默认监听
`127.0.0.1:7437`，拒绝非 loopback 地址。协议版本为 v2。

未设置 `CYAN_CONFIG` 时，配置优先级为：默认值、`~/.cyan/config.toml`、启动目录
`.cyan/config.toml`、`.env`、系统环境变量；设置后只读取指定 TOML，再应用环境变量。常用
变量为 `CYAN_CONFIG`、`CYAN_HOST`、`CYAN_PORT`、`CYAN_LOG_LEVEL`、`CYAN_LOG_FILE`、
`CYAN_LOG_FORMAT`、`CYAN_LLM_DEFAULT_MODEL`、`CYAN_TRACE_ENABLED` 和 `CYAN_TRACE_FILE`。

daemon 只在启动时解析自身配置。之后从其他目录启动的 TUI 会复用它；若依赖另一项目的
daemon/LLM 配置，应先停止并在新目录重启。Smoke 配置始终从 Job workspace 读取：

```toml
[incident.smoke]
argv = ["python", "smoke_train.py", "--steps", "2"]
timeout_s = 300
```

## 诊断文件

```bash
tail -f ~/.cyan/logs/core.log
find ~/.cyan/jobs -maxdepth 4 -type f
```

- `launch.json`：精确 argv/cwd/env，权限 `0600`，不得复制到 issue；
- `attempts/*/{stdout,stderr}.log`：完整原始输出；
- `failure.json`：失败时固定的 capsule；
- `incidents/<incident_id>/incident.json`：Incident 当前快照，包含诊断、proposal、审批回执和验证结果；
- `incidents/<incident_id>/proposal.diff`：存在候选补丁时才生成；
- `incidents/<incident_id>/smoke.{stdout,stderr}.log`：运行 smoke 时才生成；
- `incidents/<incident_id>/runs/<run_id>/{run.json,events.jsonl}`：单轮有限指标和摘要事件；
- `traces/daemon.jsonl`：脱敏后的结构化 trace 摘要。

## Incident 状态

| 状态 | 含义 |
|---|---|
| `diagnosing` | 只读调查 |
| `awaiting_approval` | diagnosis 与 preflight-valid diff 已就绪 |
| `smoke_running` | 运行用户声明的 smoke |
| `retry_running` | 用原命令最终验证 |
| `resolved` / `rejected` | 原命令成功 / 用户拒绝 |
| `stale` | workspace 或 smoke 配置变化，未应用 |
| `rollback_blocked` | 文件变化，自动回滚被阻止 |
| `unresolved` | 没有可信 proposal 或验证未完成 |

恢复遗留 smoke 时，只有 PID、启动身份、session leader 和进程组 leader 全部匹配才会终止。

## 当前运维限制

- JobStore 位于全局 `~/.cyan/jobs`；TUI 只自动恢复当前 workspace，`/jobs` 可显式选择其他任务。
- 附着已有 Attempt 时显示每个 stream 最后 32 KiB 并从该 byte cursor 继续；完整日志保留在磁盘。
- daemon 配置不随 TUI workspace 切换。

## 开发检查

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/ -v
uv run python scripts/gen_protocol_doc.py --check
```
