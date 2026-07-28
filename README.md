# cyan

cyan 是一个面向本地机器学习训练任务的 Incident Agent：训练进程正常运行时不调用
LLM；只有真实进程意外失败后，daemon 才自动唤醒一个受限 Agent，基于完整落盘日志和
当前仓库证据给出诊断与补丁。用户只审批一次，harness 负责校验补丁、可选执行 smoke
verifier，并用原命令完成最终验证。

它处理的是“训练任务崩溃后的恢复”，不是论文检索、指标优化、超参数搜索或
AutoResearch。

## 快速开始

要求 Python 3.12。
以下流程假设安装生成的 `cyan` 入口已在 `PATH` 中；仓库开发环境可直接使用
`/path/to/Zgent/.venv/bin/cyan`。

```bash
uv sync
cp .env.example .env
cd /path/to/your/ml-project
cyan
```

在 TUI 中输入：

```text
/monitor
```

普通聊天输入 `/` 会显示本地 TUI 命令和可用 Agent skills；使用上下键选择，按 Enter
回填。普通 Agent 调用受控工具时，TUI 会显示聚焦的权限选择器，可用上下键和 Enter，
或直接按 `y/a/n/d` 选择一次允许、始终允许、拒绝或始终拒绝。

粘贴一条训练启动命令并按 Enter。cyan 会确定性解析引号、反斜杠续行和开头的环境变量
覆盖，展示 cwd、可执行文件、argv、覆盖项和显式配置路径；确认无误后输入：

```text
/start
```

随后 cyan 会：

1. 在需要时自动启动 `cyan-core`，普通聊天和训练监视共用当前 TUI；
2. 以当前目录作为不可变 `workspace_root`，由 harness 启动真实 argv；
3. 持续展示命令、cwd、状态及 stdout/stderr；
4. 在非零退出或非用户信号后自动创建只读 Incident Agent；
5. 展示结构化根因、稳定证据引用和完整 proposed diff；
6. 在上下文选择器中批准或拒绝补丁，并可选择先运行用户声明的 smoke verifier；
7. 用原 argv、cwd 和环境重跑；只有真实退出码为 0 才标记 `resolved`。

重新运行 `cyan` 会新建普通聊天，并自动恢复唯一的运行中或待处理 Job/Incident；
存在多个任务时保持聊天输入可用，输入 `/jobs` 才打开任务选择器。`Ctrl+Q` 只分离
TUI，不会停止训练任务或关闭 Incident session。`cyan watch -- python train.py ...`
仍作为兼容入口保留。输入 `/monitor` 会跳过历史终态 Job/Incident；只有当前附着的
训练进程仍处于 `starting/running` 时才要求先使用运行期的 `Cancel training process`
上下文动作。

训练命令不会交给 Agent 解释，也不会通过 shell 执行。管道、重定向、后台执行和命令
替换会被拒绝；复杂启动逻辑应先写入脚本，再粘贴 `bash scripts/train_local.sh`。v1 不会
侦测用户在另一个终端自行启动的进程。

## 可选 smoke verifier

smoke 是用户预先声明的短验证，不由 Agent 生成或修改。在项目根目录创建：

```toml
# .cyan/config.toml
[incident.smoke]
argv = ["python", "smoke_train.py", "--steps", "2"]
timeout_s = 300
```

proposal 通过只读 Git preflight 后，TUI 会显示聚焦的上下文选择器：

- `Approve and run smoke`：批准补丁，先 smoke，再运行完整原命令；
- `Approve without smoke`：批准补丁，跳过 smoke，直接运行完整原命令；
- `Reject patch`：拒绝补丁。

选择器支持鼠标点击或上下键和 Enter；它们不是 slash command，也没有全局单键绑定。

smoke 与原任务使用同一个 workspace 和环境，但不是安全沙盒。它只按退出码判定：
`0` 表示通过。smoke 失败时，cyan 只会在补丁目标仍保持 apply 后哈希时自动反向应用，
避免覆盖用户并发修改；随后恢复同一个只读 Incident session 继续调查。cyan 会持久化正在
运行的 smoke 进程身份；daemon 重启时只有在 PID、启动身份、session 和进程组均匹配后
才终止遗留 verifier，然后把 Incident 恢复为 `unresolved`。

## 为什么这里需要 Agent

普通脚本可以捕获退出码，但故障恢复通常还需要跨日志、配置、调用栈和源码做动态取证。
cyan 把 Agent 放在最有价值的时间点：失败发生后自动唤醒，而不是在训练期间持续消耗
context。

- 训练阶段：只监管进程并无损保存日志，不创建 LLM context。
- 失败阶段：从固定 failure capsule 启动新的 Incident context。
- 调查阶段：Agent 只能读当前 workspace、检索有界日志并写 Incident artifact。
- 修改阶段：Agent 只提交单文件精确 SEARCH/REPLACE；harness 从真实文件生成 diff，且只有
  一次显式批准后才能应用。
- 验证阶段：smoke 只是可选短门禁，最终真值始终是原命令的真实退出状态。

## 评测导出

用标准库脚本从真实落盘 artifact 导出逐 Incident 与聚合指标：

```bash
uv run python scripts/evaluate_incidents.py --pretty > cyan-results.json
```

可用 `--jobs-root` 和 `--sessions-root` 指向隔离的评测目录。输出包含状态、修复结果、
failure-to-diagnosis/terminal 延迟、证据读取字节、token 和工具调用，并把损坏或未完成
artifact 列入 `skipped`。做 cyan、tail-only 和手动通用 Agent 对照时，应对同一故障集在
外部记录各组时间戳与结果，再把本文件作为 cyan 组的可复核观测数据；脚本不会运行 LLM
或伪造 baseline。根因/证据正确率必须由预先定义的外部 ground truth 和盲评标注计算，
不能从退出码或这些 artifact 自动推断。

## 安全与证据边界

自动调查阶段只注册以下工具：

- `read_file`
- `list_dir`
- `search_text`
- `read_job_log`
- `submit_diagnosis`
- `propose_patch`

不会注册 `write_file`、通用 `bash`、MCP、Skill、TaskManager 或 subagent。所有 workspace
工具拒绝绝对路径、`..` 和 symlink 越界。Incident 也不会加载全局/项目 context、notes
或项目 MCP 配置，安全约束和 failure capsule 每轮都从持久化元数据重建。

每次日志读取最多 32 KiB，每轮 Agent 的日志证据预算为 256 KiB。诊断必须携带稳定
byte-range 或 `path@sha256#line` 引用。完整 stdout/stderr 只保存一次；Incident 的
`events.jsonl` 不重复保存逐 token 流或完整工具输出。daemon trace 对日志、补丁、工具
输入输出、token 和用户文本同样只保留字段名与体积摘要。

补丁应用前会重新检查：

- 当前目录确实是 Git worktree 根；
- SEARCH 在对应当前文件中精确且唯一，目标是已有、不超过 1 MiB 的 UTF-8 文本文件；
- harness 生成的 diff 只修改该文件，不包含 create/delete、rename/copy、binary、
  symlink、submodule、`.git` 或 verifier 配置；
- proposal 的 patch hash、目标路径和原文件 SHA-256 未变化；
- `git apply --check` 成功。

非 Git 项目仍可诊断和审阅 diff，但 TUI 禁止一键应用。

## 运行架构

```text
cyan / TUI
    │  TCP loopback + NDJSON + JSON-RPC 2.0
cyan-core
    ├── JobSupervisor ── real process group
    ├── JobStore ─────── raw logs + immutable launch/failure evidence
    ├── IncidentCoordinator
    │   ├── read-only AgentRunner / ToolRegistry
    │   ├── PatchService
    │   └── optional SmokeExecutor
    └── EventBus ─────── bounded client subscriptions
```

daemon 是长任务的唯一所有者。TUI 通过 byte offset 增量读取原始日志，高频日志不会进入
EventBus；每个订阅有独立有界队列，慢客户端不能反压训练日志落盘。

持久化布局：

```text
~/.cyan/jobs/<job_id>/
├── job.json
├── events.jsonl
├── launch.json
├── attempts/<attempt_id>/
│   ├── attempt.json
│   ├── stdout.log
│   ├── stderr.log
│   └── failure.json
└── incidents/<incident_id>/
    ├── incident.json
    ├── diagnosis.json
    ├── proposal.diff
    ├── proposal.json
    ├── apply.json
    ├── smoke-execution.json
    ├── smoke.json
    ├── smoke.stdout.log
    └── smoke.stderr.log
```

`launch.json` 权限为 `0600`，包含精确重跑信息，但不会暴露给 Agent 或普通 RPC。daemon
只允许监听 loopback 地址；v1 不启动配置中的 MCP server。

## 非目标

v1 明确不做：

- 论文搜索和 Research Skill；
- loss/accuracy 优化、实验搜索和 keep/revert；
- 自动修改后无审批运行；
- Web dashboard 或 VS Code 插件；
- SQLite、Redis、消息队列、向量数据库或工作流引擎。

## 开发验证

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/unit -v
uv run pytest tests/integration -v
uv run python scripts/gen_protocol_doc.py --check
```

Incident 集成测试使用真实临时 Git 仓库、真实子进程、真实 `git apply/reverse` 和真实 smoke
命令；只有外部 LLM 被确定性 provider 替代。

协议见 [WIRE_PROTOCOL.md](WIRE_PROTOCOL.md)，运维说明见 [RUNBOOK.md](RUNBOOK.md)。
