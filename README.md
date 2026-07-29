# cyan

cyan 是一个面向本地机器学习训练崩溃的 Incident Agent。训练正常运行时，daemon 只监管
真实进程并持久化 stdout/stderr，不调用 LLM；进程非零退出后才唤醒受限 Agent，基于日志和
当前仓库证据给出诊断与可选补丁。补丁必须经过用户明确批准，并由原训练命令完成最终验证。

cyan 不做论文检索、指标优化、超参数搜索或 AutoResearch。

## 快速开始

要求 Python 3.12。仓库开发环境可使用：

```bash
uv sync
cp .env.example .env
cd /path/to/your/ml-project
/path/to/Zgent/.venv/bin/cyan
```

在 TUI 中：

```text
/monitor
# 粘贴一条训练命令并提交
/start
```

cyan 会先展示 cwd、可执行文件、最终 argv、环境覆盖项和识别到的配置路径；`/start`
确认后才启动训练。训练失败后的 diagnosis、证据和 proposed diff 会显示在同一 TUI，
批准、拒绝和取消训练使用当前状态下出现的上下文选择器。

本地命令只有：

```text
/monitor
/start
/jobs
/incident <text>
/help
```

输入 `/` 可用上下键和 Enter 补全本地命令与普通 Agent skills。普通聊天工具权限通过独立
选择器处理，不会扩大 Incident Agent 的只读权限。

`Ctrl+Q` 只分离 TUI，不停止训练。`cyan watch -- <argv>` 是兼容入口。

## 启动边界

训练命令由本地 harness 解析，不交给 Agent，也不经 shell 执行。支持引号、反斜杠续行和
命令开头的 `KEY=VALUE`；拒绝管道、重定向、后台执行和命令替换。复杂逻辑应封装为脚本，
例如：

```text
bash scripts/train_local.sh
```

cyan 只监视由自身启动的非交互进程，不附着用户在其他终端自行启动的任务。daemon 保存精确
argv、cwd 和环境用于重跑；成为父进程不会主动限制训练可用的 CPU 或 GPU。

## 故障闭环

```text
真实非零退出
→ failure capsule
→ 有界只读调查
→ diagnosis
→ 可选单文件 SEARCH/REPLACE
→ harness 生成并校验 diff
→ 用户审批
→ 可选 smoke
→ 原命令最终重跑
```

Incident Agent 只注册：

```text
read_file, list_dir, search_text, read_job_log, submit_diagnosis, propose_patch
```

它不能运行 shell、写工作区、启动 MCP/Skill/subagent 或读取全局 context 和 session notes。
每轮最多 12 steps，日志证据预算为 256 KiB。

补丁仅支持对一个已有、不超过 1 MiB 的 UTF-8 文件做一次精确唯一替换。Harness 校验证据
引用、当前文件 SHA-256、Git root、submodule 边界和 `git apply --check`。批准前只写私有
Incident artifact；非 Git workspace 可以诊断和审阅 diff，但不能一键应用。

## 可选 smoke verifier

在训练项目根目录创建：

```toml
[incident.smoke]
argv = ["python", "smoke_train.py", "--steps", "2"]
timeout_s = 300
```

smoke 由用户声明，不由 Agent 生成或修改。smoke 失败时，cyan 只在目标仍保持 apply 后哈希
时反向应用补丁；最终真值始终是原训练命令的真实退出码。

## 架构

```text
cyan / cyan-tui
    │ TCP loopback + NDJSON + JSON-RPC 2.0
cyan-core
    ├── JobSupervisor ── real process groups
    ├── file-backed JobStore
    ├── IncidentCoordinator
    ├── AgentRunner / ToolRegistry
    └── EventBus / SessionManager
```

原始进程输出只写 attempt 日志；TUI 按 byte cursor 读取。`launch.json` 为 `0600`，不会通过
RPC 返回或暴露给 Agent。附着已有 Attempt 时显示有界日志尾部并从该 cursor 继续，完整日志
仍保留在磁盘。IPC 客户端使用独立有界队列，慢客户端不会阻塞训练日志落盘。daemon 配置在
启动时固定；切换依赖不同项目配置的 workspace 前需要重启 daemon。

## 当前验证范围

当前本地 CPU/PyTorch Pilot 的最新回归结果为：

- 非零退出侦测 18/18；
- 根因与证据正确 18/18，长日志 6/6；
- 可修补病例首次批准后原命令成功 11/12；
- 非代码故障 6/6 未产生 proposal；
- 批准前 tracked 写入和不可应用 proposal 到达审批均为 0。

结果只证明 macOS、CPU、PyTorch 和明确异常退出场景，不外推到 CUDA、NCCL、多机训练、
OOM、进程挂死、系统休眠或静默收敛故障。完整方法和限制见
[USER_TEST_REPORT.md](USER_TEST_REPORT.md)。

## 开发验证

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/unit -v
uv run pytest tests/integration -v
uv run pytest tests/ -v
uv run python scripts/gen_protocol_doc.py --check
```

协议见 [WIRE_PROTOCOL.md](WIRE_PROTOCOL.md)，运维说明见 [RUNBOOK.md](RUNBOOK.md)，开发约束
见 [AGENTS.md](AGENTS.md)。
