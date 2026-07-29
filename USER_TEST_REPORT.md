# cyan 真实 ML 项目 Pilot 测试报告

## 当前结论

当前工作区在本地 CPU/PyTorch 范围内完成一轮 9-case 真实 LLM Pilot：根因与证据判断
9/9 正确，6 个可修补病例首次批准后全部由原命令验证成功，3 个参数或路径故障均停在操作
建议且没有 proposal，3 个超过 5 MiB 的长日志病例全部正确处理。

该结果支持 cyan 作为受管的本地训练崩溃恢复 Agent，不支持将结论外推到通用 ML 故障、
CUDA、分布式训练或静默质量问题。

## 版本与环境

| 项目 | 版本 |
|---|---|
| Pilot 测试基线 | `26a8b12dd7344c8b383540f55a494d0c7e09d976` + 当前工作区 |
| 日期 | 2026-07-29 |
| 主机 | macOS 26.5.2，Apple arm64，CPU-only |
| Python / PyTorch | 3.12.13 / 2.13.0，三个独立 `uv` 环境 |
| Agent 模型 | `deepseek-v4-flash` |
| FastestDet 上游 / fixture | `50473cd155cb088aa4a99e64ff6a4b3c24fa07e1` / `004715b4bfc70a850ffac5cb313fede8b4ac08e9` |
| PyTorch examples 上游 | `acc295dc7b90714f1bf47f06004fc19a7fe235c4` |
| MNIST / WLM fixture | `1b4c5d0f165d91fd5f73b821c2796ffc8b346a37` / `d9c7f5795032c22b84e1dff95ce2219d88eef82f` |

密钥未写入测试产物。

## 测试设计

覆盖 FastestDet、PyTorch MNIST 和 word language model 三个 workload，每个 workload 执行
三类冻结故障：

1. 可由一个最小代码替换修复的故障；
2. 数据、路径、环境或启动参数故障；
3. stdout 超过 5 MiB、根因证据不在日志尾部的晚期崩溃。

9 个故障各运行一次，共 9 个真实 Incident。测试使用独立 HOME、JobStore、端口、真实 Git
仓库、真实训练子进程和 `deepseek-v4-flash`；Agent 不接触 ground truth。6 个代码病例在
proposal 到达后批准，3 个非代码病例不作决定。本轮针对核心 LLM 闭环，未重复 TUI 路径、
正常训练、tail-only 或性能测试。

## 最新结果

| Workload | 诊断正确 | 长日志正确 | 首次批准后成功 | 非代码病例无 proposal |
|---|---:|---:|---:|---:|
| FastestDet | 3/3 | 1/1 | 2/2 | 1/1 |
| MNIST | 3/3 | 1/1 | 2/2 | 1/1 |
| Word language model | 3/3 | 1/1 | 2/2 | 1/1 |
| **合计** | **9/9** | **3/3** | **6/6** | **3/3** |

| 验收项 | 实测 |
|---|---:|
| 非零退出侦测 | 9/9 |
| 根因与 evidence 正确 | 9/9 |
| 长日志正确 | 3/3 |
| 首次批准后原命令成功 | 6/6 |
| 非代码故障不产生 proposal | 3/3 |
| proposal 修改范围正确 | 6/6 |
| proposal 与当前 diagnosis 自动关联 | 6/6 |
| 不可应用 proposal 到达审批 | 0 |
| 批准前 tracked 写入 | 0 |
| `launch.json` 权限 | 9/9 为 `0600` |
| RPC 暴露环境或 `launch.json` | 0/9 |
| 测试产物包含 API 密钥 | 0 |

诊断中位耗时为 19.7 秒，终态中位耗时为 29.8 秒；9 次运行共调用工具 61 次，记录
input/output token 92,480/24,640，Agent 主动读取日志证据 67,426 bytes。

此前失败的 FastestDet 长日志病例本轮正确生成 `train.py` 最小补丁并 `resolved`；Proposal
中的 diagnosis ID 与 Harness 当前 diagnosis 一致，验证了自动绑定修复。

## 冻结基线与限制

以下冻结数据本轮未重新运行，因为四项修复不改变训练进程监督热路径：

- tail-only 正确 11/18；
- 正常训练误触发 0/15；
- FastestDet、MNIST、WLM 中位运行时开销分别为 +56.4%、-0.5%、-0.8%。

FastestDet 性能数据来自约 3.6 秒的极短 fixture，其中约 2 秒为固定开销，不能外推到长时
训练。本测试还未覆盖 CUDA、NCCL、多机、OOM、DataLoader 多进程、磁盘耗尽、进程挂死、
系统休眠、交互式训练或静默收敛故障。长日志用例验证全文检索，不等价于跨一小时、多阶段、
模糊分散证据的 long-memory 推理。

## 产物与代码验收

隔离产物：

```text
/private/tmp/cyan-llm-pilot-cOJSMG/
/private/tmp/cyan-llm-pilot-cOJSMG/home/.cyan/
```

当前工作区重新验收结果：

```text
Ruff: passed
mypy: passed
WIRE_PROTOCOL.md: up to date
unit: 425 passed
integration: 18 passed
pytest: 443 passed in 40.44s
```

## VS Code Alpha 验收

2026-07-29 增加的薄型 VS Code 客户端未修改 Incident Agent、进程监督或补丁应用策略，
因此没有重跑上述 9-case LLM Pilot。新增验收覆盖：

- TypeScript 编译与 7 个纯 UI、cursor、Trust 和上下文动作测试；
- 真实 VS Code Extension Host 启动、daemon 连接、命令注册和 snapshot refresh；
- 真实 daemon、真实训练子进程的 `launch.preview → launch.start → job.read_log`；
- VSIX 成功打包；关闭扩展只断开客户端；
- 5 组交替运行中，无日志读者中位 0.8126 秒，300 ms cursor 轮询中位 0.8076 秒，
  观测差值 -0.61%，未发现可测的训练开销。

性能数字来自 0.75 秒 sleep fixture，只用于排除轮询造成的明显固定开销，不外推到真实模型
吞吐或 GPU 利用率。
