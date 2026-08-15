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
- 人工动作：按 diagnosis 完成操作后选择 `Recheck workflow`

选择器支持鼠标或上下键和 Enter。`Esc` 返回输入框，`Ctrl+Q` 只分离 TUI。

## daemon 与配置

正常流程自动启动 daemon。开发诊断命令：

```bash
uv run cyan core status
uv run cyan core start
uv run cyan core stop
uv run cyan ping
```

daemon 默认监听 `127.0.0.1:7437`，拒绝非 loopback 地址，且当前实现不启动配置中的 MCP server。

未设置 `CYAN_CONFIG` 时，配置优先级为：默认值、`~/.cyan/config.toml`、启动目录
`.cyan/config.toml`、`.env`、系统环境变量；设置后只读取指定 TOML，再应用环境变量。常用
变量为 `CYAN_CONFIG`、`CYAN_HOST`、`CYAN_PORT`、`CYAN_LOG_LEVEL`、`CYAN_LOG_FILE`、
`CYAN_LLM_DEFAULT_MODEL` 和 `CYAN_PERMISSION_TIMEOUT_S`。

daemon 只在启动时解析自身配置。之后从其他目录启动的 TUI 会复用它；若依赖另一项目的
daemon/LLM 配置，应先停止并在新目录重启。Smoke 配置始终从 Job workspace 读取：

```toml
[incident.smoke]
argv = ["python", "smoke_train.py", "--steps", "2"]
timeout_s = 300
```

可选 `.cyan/workflow.toml` 在 preview/start 时读取并冻结；retry 不重新读取磁盘版本。主命令
和 checks 都是 trusted user-owned executables。修改 Contract 后必须重新 preview，不能通过
修改 Contract 修复一个已开始的 Incident。

## 诊断文件

```bash
tail -f ~/.cyan/logs/core.log
find ~/.cyan/jobs -maxdepth 4 -type f
```

- `launch.json`：精确 argv/cwd/env/Contract snapshot，权限 `0600`，不得复制到 issue；
- `attempts/*/{stdout,stderr}.log`：完整原始输出；
- `failure.json`：失败时固定的 capsule；
- `incidents/*/{diagnosis.json,proposal.diff}`：诊断和候选补丁；
- `incidents/*/smoke-execution.json`：verifier PID 与进程身份；
- `traces/daemon.jsonl`：脱敏后的结构化 trace 摘要。

## Incident 状态

| 状态 | 含义 |
|---|---|
| `diagnosing` | 只读调查 |
| `awaiting_approval` | diagnosis 与 preflight-valid diff 已就绪 |
| `action_required` | 等待用户完成结构化人工动作并显式重检 |
| `smoke_running` | 运行用户声明的 smoke |
| `retry_running` | 用 frozen full workflow 最终验证 |
| `resolved` / `rejected` | 完整 workflow 成功 / 用户拒绝 |
| `stale` | workspace 或 smoke 配置变化，未应用 |
| `rollback_blocked` | 文件变化，自动回滚被阻止 |
| `unresolved` | 没有可信 proposal 或验证未完成 |

恢复遗留 smoke 时，只有 PID、启动身份、session leader 和进程组 leader 全部匹配才会终止。

## 当前运维限制

- JobStore 位于全局 `~/.cyan/jobs`；TUI 只自动恢复当前 workspace，`/jobs` 可显式选择其他任务。
- 附着已有 Attempt 时显示每个 stream 最后 32 KiB 并从该 byte cursor 继续；完整日志保留在磁盘。
- daemon 配置不随 TUI workspace 切换。
- artifacts 只支持 workspace-local path；`fresh=true` 只判断 metadata 更新，不判断内容正确。
- Contract 和 checks 不提供 sandbox、DAG、外部 root 或局部阶段重跑。

## Evidence Retrieval Benchmark

Evidence Retrieval Benchmark 的快速离线检查：

```bash
BENCH_ROOT=/private/tmp/cyan-benchmark-ci
uv run python scripts/benchmark_evidence.py prepare --corpus "$BENCH_ROOT" --profile ci
uv run python scripts/benchmark_evidence.py validate --corpus "$BENCH_ROOT"
uv run python scripts/benchmark_evidence.py run \
  --corpus "$BENCH_ROOT" --method oracle --output /private/tmp/oracle.run.json
uv run python scripts/benchmark_evidence.py score \
  --corpus "$BENCH_ROOT" --run /private/tmp/oracle.run.json \
  --output /private/tmp/oracle.scores.json
```

真实48-case Core 需先运行 `uv sync --group benchmark`。检索 `run` 默认为每个 Case 启动独立
worker，以记录不受前序 Case 污染的 peak RSS；仅开发调试可使用 `--no-isolation`。

正式 Core 报告必须通过 `validate --require-complete-core`。该检查要求恰好 60 个 Core Case，
其中12个具有公开 commit 对或 issue + revision 对，且所有 test Gold 已完成两名不同 reviewer
的一致审核；
`--allow-incomplete-core` 生成的开发集不得用于正式结论。人工流程使用
`gold-review-export/import` 和 `review-export/import`，评审文件不得包含模型密钥。

Defects4ML 重放先按 `benchmarks/evidence_retrieval/docker/` 构建四个固定 x86 镜像，再运行：

```bash
uv run python scripts/benchmark_evidence.py history-prepare \
  --archive /path/to/bugs.zip --output /path/to/bundles --work-dir /path/to/work
```

命令断网运行并保存每个候选日志；不足12个时返回非零且不发布 bundle。

## 开发检查

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/ -v
uv run python scripts/gen_protocol_doc.py --check
```
