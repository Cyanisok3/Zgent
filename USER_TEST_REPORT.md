# Cyan 当前重构验收报告

本报告记录 Cyan 模块收敛后的可重复工程验收，不把未重新运行的真实模型 Pilot 结果冒充为
当前版本指标。

## 已验证范围

- Agent 内核只执行 `LLM → tool → observe → stop`，Incident profile 固定六个工具。
- JobSupervisor 仍负责真实训练子进程、进程组、stdout/stderr、失败胶囊和原命令重跑。
- Incident Runtime 负责 selector、有限上下文和单轮 run；Coordinator 独占 FSM、审批、Patch、Smoke、重跑和恢复。
- Proposal、审批回执、Smoke 状态和 Run 摘要使用合并后的 Incident 持久化布局。
- Wire Protocol v2 包含 `incident.follow_up`，旧通用 Agent/Session/Permission 表面不再注册。
- Evidence Selector 在超过 5 MiB 的真实日志上分块扫描，初始证据限制为 32 KiB。

## 当前本地结果

| 检查 | 结果 |
|---|---:|
| Ruff | 通过 |
| mypy | 通过（71 个源文件） |
| 单元测试 | 263 passed |
| 集成测试 | 13 passed |
| 全量 Python 测试 | 276 passed |
| 协议文档生成检查 | 通过 |
| VS Code TypeScript lint | 通过 |
| VS Code 单元测试 | 7 passed |
| VSIX 打包 | 通过 |

VS Code Extension Host 测试未在当前机器执行，因为预期的本地 Visual Studio Code 应用路径
不存在；安装 VS Code 后可直接运行 `npm run test:extension` 重试。

## 仍未宣称支持

当前版本仍不覆盖 CUDA/NCCL、多机训练、挂死或静默收敛故障，也不引入通用 Chat、MCP、多
Agent、Sandbox 或实验循环。真实模型 Pilot 需要在独立环境中按当前协议和持久化布局重新执行。
