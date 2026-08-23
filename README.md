# cyan

cyan 是一个面向本地机器学习训练崩溃的 Incident Agent。训练正常运行时，daemon 只监管真实
进程并持久化 stdout/stderr；进程非零退出后才唤醒受限 Agent，基于日志和当前仓库证据给出
诊断与可选补丁。补丁必须经过用户明确批准，并由原训练命令完成最终验证。

cyan 不做论文检索、指标优化、超参数搜索、实验循环或通用编码 Agent。

## 快速开始

要求 Python 3.12，并在本地安装 `cyan`：

```bash
cp .env.example .env
cd /path/to/your/ml-project
cyan
```

在 TUI 中使用：

```text
/monitor
# 粘贴一条训练命令并提交
/start
```

TUI 会先展示 cwd、可执行文件、最终 argv、环境覆盖项和配置路径；确认后才启动训练。失败后
的 diagnosis、证据和 proposed diff 在同一界面展示，批准、拒绝、smoke 选择和取消训练使用
当前状态下出现的上下文控件。

本地命令只有：

```text
/monitor
/start
/jobs
/incident <text>
/help
```

默认状态的普通文本不会发送给 Agent；追问必须显式使用 `/incident <text>`。`Ctrl+Q` 只分离
TUI，不停止 daemon-owned 训练。

## CLI 与 VS Code

```text
cyan                         # 唯一正常用户入口，启动 TUI
cyan --version
cyan core start [--json]     # daemon 发现/后台启动
cyan core stop               # 通过 RPC 安全停止 daemon
cyan-core                    # 内部 daemon 进程入口
```

`watch`、`chat`、`run`、`ping`、`trace`、`core status` 和独立 `cyan-tui` 入口已删除。VS Code
Alpha 是可选薄客户端：只负责预览、日志、诊断、原生 diff 和上下文操作，训练、Incident、补丁
与重跑仍由 Python daemon 负责。

## 启动边界

训练命令由本地 harness 解析，不交给 Agent，也不经 shell 执行。支持引号、反斜杠续行和命令
开头的 `KEY=VALUE`；拒绝管道、重定向、后台执行和命令替换。复杂逻辑应封装为脚本，例如：

```text
bash scripts/train_local.sh
```

daemon 保存精确 argv、cwd 和环境用于重跑，只监视由自身启动的非交互进程。

## 故障闭环

```text
真实非零退出
→ Failure Capsule
→ Evidence Selector（最多 32 KiB 初始证据）
→ 六工具只读 Incident Agent
→ Diagnosis / 单文件 Proposal
→ 用户审批
→ PatchService
→ 可选 Smoke
→ 原命令最终重跑
```

Incident Agent 只注册：

```text
read_file, list_dir, search_text, read_job_log, submit_diagnosis, propose_patch
```

它不能运行 shell、写工作区、加载全局/项目 context、读取旧 Session、启动 MCP/Skill/subagent
或执行 Sandbox。每轮最多 12 steps；serialized system、messages 和 tool schemas 的总输入不
超过 128 KiB，初始 selector 证据不超过 32 KiB。

补丁仅支持对一个已有、不超过 1 MiB 的 UTF-8 文件做一次精确唯一替换。批准前只写 Incident
artifact；非 Git workspace 可以诊断和审阅 diff，但不能一键应用。

## 持久化布局

```text
~/.cyan/jobs/<job_id>/incidents/<incident_id>/
├── incident.json
├── proposal.diff              # 有 proposal 时才存在
├── smoke.stdout.log           # 运行 smoke 时才存在
├── smoke.stderr.log
└── runs/<run_id>/{run.json,events.jsonl}
```

`incident.json` 是当前快照，嵌入 FSM 状态、diagnosis、proposal metadata、apply receipt 和
smoke 结果。`run.json` 只保存本轮指令、证据引用、字节指标和结构化结果；完整训练日志、完整
tool output 和逐 token 文本不会重复写入。旧 `~/.cyan/sessions` 不迁移、不读取。

## 可选 Smoke Verifier

在训练项目根目录创建：

```toml
[incident.smoke]
argv = ["python", "smoke_train.py", "--steps", "2"]
timeout_s = 300
```

Smoke 由用户声明，Agent 不能修改。Smoke 失败时，仅在目标仍保持 apply 后哈希时反向应用补丁；
最终真值始终是原训练命令的真实退出码。

## 架构

```text
cyan / cyan VS Code
    │ TCP loopback + NDJSON + JSON-RPC 2.0
cyan-core
    ├── service/    daemon、协议、传输
    ├── agent/      AgentRunner、AgentLoop、EventBus、ToolRegistry
    └── training/   JobSupervisor、JobStore、IncidentCoordinator/Runtime
```

协议版本为 `2`。变更协议模型后运行：

```bash
uv run python scripts/gen_protocol_doc.py
uv run python scripts/gen_protocol_doc.py --check
```

## 开发验证

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/unit -v
uv run pytest tests/integration -v
uv run pytest tests/ -v
uv run python scripts/gen_protocol_doc.py --check
cd vscode && npm run lint && npm run test:unit && npm run test:extension && npm run package
```

开发约束见 [AGENTS.md](AGENTS.md)，协议见 [WIRE_PROTOCOL.md](WIRE_PROTOCOL.md)，VS Code 说明见
[vscode/README.md](vscode/README.md)。
