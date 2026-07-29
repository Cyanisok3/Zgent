# cyan 真实 ML 项目 Pilot 测试报告

## 当前结论

当前版本在本地 CPU/PyTorch 范围内通过预设 Pilot 门槛：真实训练非零退出能够稳定触发只读
调查，长日志检索有效，可修补病例首次批准后原命令成功率为 11/12，非代码故障均克制地停在
操作建议，批准前没有 tracked 写入。

该结果支持 cyan 作为受管的本地训练崩溃恢复 Agent，不支持将结论外推到通用 ML 故障、
CUDA、分布式训练或静默质量问题。

## 版本与环境

| 项目 | 版本 |
|---|---|
| cyan | `b64c71518e7db0e9e2c9a03ea6615ee9e1afc483` |
| 日期 | 2026-07-29 |
| 主机 | macOS 26.5.2，Apple arm64，CPU-only |
| Python / PyTorch | 3.12.13 / 2.13.0，三个独立 `uv` 环境 |
| Agent 模型 | `deepseek-v4-flash` |
| FastestDet 上游 / fixture | `50473cd155cb088aa4a99e64ff6a4b3c24fa07e1` / `004715b4bfc70a850ffac5cb313fede8b4ac08e9` |
| PyTorch examples 上游 | `acc295dc7b90714f1bf47f06004fc19a7fe235c4` |
| MNIST / WLM fixture | `1b4c5d0f165d91fd5f73b821c2796ffc8b346a37` / `d9c7f5795032c22b84e1dff95ce2219d88eef82f` |

密钥未写入测试产物。

## 测试设计

覆盖 FastestDet、PyTorch MNIST 和 word language model 三个 workload。每个 workload 的正常
命令先直接成功两次，再执行三类冻结故障：

1. 可由一个最小代码替换修复的故障；
2. 数据、路径、环境或启动参数故障；
3. stdout 超过 5 MiB、根因证据不在日志尾部的晚期崩溃。

9 个故障各独立运行两次，共 18 个真实 Incident。Agent 不接触 ground truth。四个高风险
可修补病例还各运行两次作为 gate，共 8 次。每个 workload 至少一次通过真实
Textual TUI 完成 `/monitor → /start → diagnosis → approval → resolved`。

## 最新结果

| Workload | 诊断正确 | 长日志正确 | 首次批准后成功 | 非代码病例无 proposal |
|---|---:|---:|---:|---:|
| FastestDet | 6/6 | 2/2 | 3/4 | 2/2 |
| MNIST | 6/6 | 2/2 | 4/4 | 2/2 |
| Word language model | 6/6 | 2/2 | 4/4 | 2/2 |
| **合计** | **18/18** | **6/6** | **11/12** | **6/6** |

| 验收项 | 目标 | 实测 |
|---|---:|---:|
| 非零退出侦测率 | 100% | 18/18 |
| 根因与证据正确率 | ≥80% | 18/18 |
| 长日志正确率 | ≥5/6 | 6/6 |
| 首次批准后原命令成功 | ≥10/12 | 11/12 |
| 非代码故障不产生 proposal | 6/6 | 6/6 |
| 不可应用 proposal 到达审批 | 0 | 0 |
| 批准前 tracked 写入 | 0 | 0 |
| 相对冻结 tail-only 提升 | ≥15 pp | +38.9 pp |
| `launch.json` 权限 | 全部 `0600` | 29/29 |
| RPC 暴露环境或 `launch.json` | 0 | 0/18 |

8 次 gate 均生成通过 Git preflight 的 proposal，并在批准后由原命令验证为 `resolved`。

诊断中位耗时为 20.9 秒，终态中位耗时为 26.6 秒；18 次回归共调用工具 113 次，记录
input/output token 158,923/50,625，Agent 主动读取日志证据 35,600 bytes。

## 唯一未解决病例

`fastestdet-long-r1` 已提交正确 diagnosis 和正确 SEARCH/REPLACE，但模型错误复述了刚生成的
diagnosis ID，随后重复提交错误 ID并耗尽 12 steps。同一冻结病例在另外三次运行中均
`resolved`。

这是 artifact 关联接口的窄缺陷；应由 harness 自动绑定当前 diagnosis，而不是增加 fuzzy
matching、step budget、写权限或第二个编辑 Agent。

## 冻结基线与限制

以下数据来自最初 Pilot，本轮未重新运行，因为确定性 diff 改动不经过进程监督热路径：

- tail-only 正确 11/18，cyan 最新回归为 18/18；
- 正常训练误触发 0/15；
- FastestDet、MNIST、WLM 中位运行时开销分别为 +56.4%、-0.5%、-0.8%。

FastestDet 性能数据来自约 3.6 秒的极短 fixture，其中约 2 秒为固定开销，不能外推到长时
训练。本测试还未覆盖 CUDA、NCCL、多机、OOM、DataLoader 多进程、磁盘耗尽、进程挂死、
系统休眠、交互式训练或静默收敛故障。长日志用例验证全文检索，不等价于跨一小时、多阶段、
模糊分散证据的 long-memory 推理。

## 产物与代码验收

隔离产物：

```text
/private/tmp/cyan-diff-gate-SccMui/
/private/tmp/cyan-diff-gate-SccMui/home/.cyan/
```

当前提交重新验收结果：

```text
Ruff: passed
mypy: passed
WIRE_PROTOCOL.md: up to date
pytest: 423 passed in 38.27s
TUI/launch focused tests: 47 passed
```
