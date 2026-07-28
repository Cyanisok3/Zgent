# cyan 真实 ML 项目 Pilot 测试报告

## 结论

cyan 已证明“真实训练进程退出后自动取证并诊断”这一垂直方向有价值：故障侦测和批准前安全边界均通过，诊断相对同模型 tail-only 基线提升 **16.7 个百分点**。但当前版本尚未达到可靠的端到端修复产品标准：总体诊断正确率为 **77.8%**，长日志病例为 **3/6**，首次批准后原命令成功率仅 **2/12（16.7%）**。主要瓶颈是补丁格式无效、部分运行未提交诊断、非代码故障误提补丁，以及历史 Incident 阻塞新 TUI 流程。

本报告只适用于本地 CPU/PyTorch pilot，不外推到 CUDA、NCCL、多机训练、OOM、系统休眠或静默收敛故障。

## 版本与环境

| 项目 | 版本 |
|---|---|
| cyan | `e21745384038e835e19ddbeff5c1a518f1b4652d` |
| 主机 | macOS 26.5.2, Apple arm64, CPU-only |
| Python / PyTorch | 3.12.13 / 2.13.0，三个独立 `uv` 环境 |
| FastestDet 上游 / fixture | `50473cd155cb088aa4a99e64ff6a4b3c24fa07e1` / `004715b4bfc70a850ffac5cb313fede8b4ac08e9` |
| PyTorch examples 上游 | `acc295dc7b90714f1bf47f06004fc19a7fe235c4` |
| MNIST / WLM fixture | `1b4c5d0f165d91fd5f73b821c2796ffc8b346a37` / `d9c7f5795032c22b84e1dff95ce2219d88eef82f` |
| Agent 模型 | 项目 `.env` 配置的 `deepseek-v4-flash`；密钥未写入测试产物 |

每个 workload 的正常命令先连续直接成功两次。9 个预先冻结 ground truth 的故障各运行两次，共 18 个真实 Incident；另运行 18 次同模型 tail-only 诊断、15 次 cyan 正常训练和 15 次直接训练。长日志病例的 stdout 分别为 5.59–6.72 MiB，均超过预设 5 MiB。

## 结果汇总

| Workload | 诊断正确 | 长日志正确 | 首次批准后成功 | 非代码病例正确克制 |
|---|---:|---:|---:|---:|
| FastestDet | 5/6 | 1/2 | 0/4 | 2/2 |
| MNIST | 5/6 | 1/2 | 1/4 | 0/2 |
| Word language model | 4/6 | 1/2 | 1/4 | 2/2 |
| **合计** | **14/18** | **3/6** | **2/12** | **4/6** |

| 验收项 | 目标 | 实测 | 结果 |
|---|---:|---:|---|
| 非零退出侦测率 | 100% | 18/18 | 通过 |
| 正常训练误触发 | 0% | 0/15 | 通过 |
| 根因与证据正确率 | ≥80% | 14/18，77.8% | 未通过 |
| 长日志正确率 | ≥5/6 | 3/6 | 未通过 |
| 首次批准后原命令成功率 | ≥75% | 2/12，16.7% | 未通过 |
| 批准前工作区写入 | 0 | 0 个 tracked 写入 | 通过 |
| 非代码故障不提无关补丁 | 100% | 4/6 | 未通过 |
| 相对 tail-only 提升 | ≥15 pp | 77.8% vs 61.1%，+16.7 pp | 通过 |
| 每个 workload 中位运行时开销 | ≤3% | +56.4% / -0.5% / -0.8% | 未通过 |
| `launch.json` 权限 | 全部 `0600` | 18/18 | 通过 |
| RPC 不暴露启动环境 | 0 次 | 0/18 | 通过 |

Incident 的诊断中位耗时为 34.2 秒，终态中位耗时为 41.0 秒；共调用工具 234 次，记录 input/output token 615,913/91,588，Agent 主动读取的日志证据共 46,195 bytes。执行了 8 次明确补丁决策（6 次批准、2 次拒绝）。一次 16.6 秒 WLM 正常训练期间，daemon 采样峰值为约 34 MiB RSS、2.1% CPU。

FastestDet 的 +56.4% 来自约 3.6 秒的极短 fixture 上约 2 秒固定开销，不能外推到小时级训练；但按本轮既定规则仍判为失败。MNIST 和 WLM 未观察到可测的运行时损失。

## 典型案例

- **成功**：MNIST 长日志病例从 5.59 MiB stdout 后的 `Target 9 is out of bounds` 回溯到九分类输出，批准最小补丁后原命令成功。
- **成功**：WLM 长日志第二次运行定位 decoder 少一个 vocabulary logit，批准后原命令成功。
- **正确克制**：FastestDet 缺失 YAML 和 WLM 缺失 corpus 路径均给出用户操作诊断，没有提代码补丁。
- **失败**：4 个方向正确的提案因 unified diff 损坏而在 `git apply --check` 阶段变为 `stale`；另有 4/18 次未产出诊断。
- **越界倾向**：FastestDet 一次同时修改直接原因文件和 `train.py`；MNIST 的两次 `--batch-size 0` 启动错误都提出了不必要的代码补丁，但未被应用。

## 缺陷分类与优先级

1. **P0，补丁闭环**：4 个正确方向的补丁是损坏 diff，直接拉低最终解决率。应先保证 `propose_patch` 产物可解析、可 `git apply --check`，失败时向用户展示明确原因，而不是扩大 Agent 权限。
2. **P0，TUI 可达性**：真实 TUI 启动后被历史可处理 Job 的 “Choose a job to attach” 页面阻塞，无法进入新的 `/monitor` 流程。本轮因此只完成了一次真实 TUI 尝试，其余病例通过相同 JSON-RPC `job.start → JobSupervisor → IncidentCoordinator` 链路运行。这是本轮测试偏差，也是实际 UX blocker。
3. **P1，诊断稳定性**：4 次运行没有提交 diagnosis；长日志只有 3/6。应优先检查 step budget、日志定位策略和终止原因，而不是立即引入常驻 subagent 或新的 memory 层。
4. **P1，补丁克制**：启动参数错误应默认输出操作建议；MNIST `batch-size=0` 的两次误补丁说明 abstain 边界仍不稳定。
5. **P2，性能与适用范围**：用至少 10 分钟的 CPU 训练重新测固定开销；之后再独立验证 CUDA、DataLoader 多进程、OOM 和信号退出，不应由本轮结果推断。

## 产品判断

当前架构在进程监督、故障胶囊、只读诊断和显式批准边界上是有效且实用的，真实日志检索也确实优于只看尾部。产品问题不在于缺少更多架构层，而在于最短主链路尚不够可靠：用户还不能稳定地从 `/monitor` 走到一个可应用且能通过原命令验证的补丁。下一轮应只修复上述两个 P0，再原样回归这 18 个病例；在最终解决率显著提高前，不建议先做长记忆 subagent 或 VS Code 包装。

## 测试产物

机器可读汇总、ground truth、18 个 Incident 结果、18 个 tail-only 结果、性能数据和资格验证日志位于：

`/private/tmp/cyan-pilot-e217-oPUULu/artifacts/`

cyan 的原始 Job/Incident/日志仍按产品机制保存在 `~/.cyan/jobs/`；对应 Job ID 可由 `artifacts/pilot-results.json` 反查。测试过程中未修改 cyan 产品代码。

## 2026-07-28 修补后回归

### 版本与范围

本节保留上方原始 Pilot 结果，并记录两阶段核心闭环修补后的独立回归。测试版本以
`c67bb255ddb33b07837cef33af5d35edaf1f40c0` 为基线，产品、测试和文档增量（不含本报告）
的 SHA-256 为 `2606c17e9ee33ef6907cc1ed27781d6ad429a1d8d9a309648359747c9f314660`。

回归继续使用原冻结 ground truth、三个原 PyTorch 环境和
`deepseek-v4-flash`。所有病例都从故障 commit 重新本地克隆，未复用旧批准流程修改过的
工作区。未重复 tail-only、15 次正常训练或性能交替测试：这些冻结结果仍分别为 11/18、
0/15 误触发和 FastestDet/MNIST/WLM 中位开销 +56.4%/-0.5%/-0.8%；本次改动未修改进程
监督热路径。

### 最终 18 次回归

| Workload | 诊断正确 | 长日志正确 | 首次批准后成功 | 非代码病例正确克制 |
|---|---:|---:|---:|---:|
| FastestDet | 6/6 | 2/2 | 3/4 | 2/2 |
| MNIST | 6/6 | 2/2 | 3/4 | 2/2 |
| Word language model | 6/6 | 2/2 | 2/4 | 2/2 |
| **合计** | **18/18** | **6/6** | **8/12** | **6/6** |

| 验收项 | 目标 | 修补后实测 | 结果 |
|---|---:|---:|---|
| 非零退出侦测率 | 100% | 18/18 | 通过 |
| 正常训练误触发 | 0% | 保留原冻结 0/15 | 通过 |
| 根因与证据正确率 | ≥80% | 18/18，100% | 通过 |
| 长日志正确率 | ≥5/6 | 6/6 | 通过 |
| 首次批准后原命令成功率 | ≥75% | 8/12，66.7% | **未通过** |
| 非代码故障不提无关补丁 | 6/6 | 6/6 | 通过 |
| 不可应用 proposal 到达审批 | 0 | 0 | 通过 |
| 批准前 tracked 写入 | 0 | 0 | 通过 |
| 相对冻结 tail-only 提升 | ≥15 pp | 100% vs 61.1%，+38.9 pp | 通过 |
| `launch.json` 权限 | 全部 `0600` | 18/18 | 通过 |
| RPC 不暴露启动环境 | 0 | 0/18 | 通过 |

诊断中位耗时为 17.6 秒，终态中位耗时为 26.6 秒；共调用工具 145 次，记录
input/output token 168,971/69,397，Agent 主动读取日志证据 36,323 bytes。所有 18 个
diagnosis 均按冻结 ground truth 人工复核，证据引用同时通过 harness 的身份和子范围校验。

### 定向与 TUI 回归

- 原先 4 个 evidence-rejection 病例现在全部提交正确 diagnosis；未增加 12-step cap。
- 原先 4 个损坏 diff 病例中，3 个在批准后通过原命令，1 个因 Agent 未在 12 步内生成
  合法 hunk 而保持 unresolved；所有损坏候选都在审批前被清除。
- 多历史任务不再阻塞新 `/monitor`，真实 daemon、真实 Git 仓库和真实子进程集成测试通过。
- FastestDet 通过真实 TUI 鼠标选择器批准并 resolved；MNIST 通过真实 TUI Enter 批准并
  resolved。
- WLM 两次真实 TUI 尝试都提交了正确 diagnosis，但没有生成 preflight-valid proposal，
  因而按设计不显示审批选择器；该 workload 的 `↑↓ + Enter` 真实 LLM 路径未完成。键盘和
  鼠标事件映射本身由 Textual DOM 测试覆盖，但不能替代这项未完成的真实 Agent 验收。

### 结论

修补后，产品定位与 diagnosis 功能已经一致：真实训练崩溃能够稳定触发只读取证，长日志
检索有效，启动参数、数据和环境故障能克制地停在操作建议，且坏补丁不再到达用户审批。
主要剩余缺陷已从“诊断与系统阻断”收敛为“LLM 生成 unified diff 的格式稳定性”：最终解决率
为 66.7%，仍不满足 75% 门槛。因此下一步应继续优化最小补丁表示或生成约束，而不是增加
step budget、写权限、常驻 subagent、memory 层或先开发 VS Code 插件。

机器结果与本轮隔离日志位于：

`/private/tmp/cyan-pilot-regression-uKXnPl/artifacts/final-regression-results.json`

最终代码验收为 Ruff 通过、mypy 通过、协议文档一致，完整 pytest
`415 passed in 38.38s`。
