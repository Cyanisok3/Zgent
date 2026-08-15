# cyan 真实 ML 项目 Pilot 测试报告

## 当前结论

当前工作区在本地 CPU/PyTorch 范围内完成一轮 9-case 真实 LLM Pilot：根因与证据判断
9/9 正确，6 个可修补病例首次批准后全部由原命令验证成功，3 个参数或路径故障均停在操作
建议且没有 proposal，3 个超过 5 MiB 的长日志病例全部正确处理。

该结果支持 cyan 原有的受管本地训练崩溃恢复闭环，不支持将结论外推到通用 ML 故障、
CUDA、分布式训练或静默质量问题。

当前实现已把 failure semantics 扩展为“退出码 + 用户声明的 Workflow Contract”，把 recovery
扩展为 `patch | operator_action | none`。Contract 解析、freshness、pre/main/post 生命周期、
零 LLM 的确定性路由、frozen retry 和客户端展示已进入自动化测试。新的 Benchmark 已实际
运行数据处理、特征转换和CPU/PyTorch workload，但没有通过生产 Incident Runtime 调用LLM；
因此下面旧9-case数字仍只代表原有明确非零退出 Pilot，不应作为 Contract 违约的Agent实测。

## Evidence Retrieval Benchmark v1

当前工作区已实现独立 Benchmark harness、隔离检索 worker、完整指标与配对 bootstrap、
Gold/Agent 双人盲审接口和可选 Agent 策略赛道，没有修改生产 Incident profile。2026-08-15
实际运行了48个 Pandas/scikit-learn/PyTorch CPU 受控 Case、35个 LogDx-CI v1.2 Case，以及
9个5/50/500 MiB LogHub压力 Case，共92 Cases。48个 Core Case 均完成失败/成功双重放，
切分为36 train / 8 dev / 4 test。

### Cyan Core 48（尚未加入12个历史故障）

|方法|R64|R128|R256|Supporting|nDCG|First B|Density|Miss|P50 ms|P95 ms|RSSΔ P95 MiB|扫描 MiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|BM25|.500|1.000|1.000|1.000|.294|61,340|.00061|.000|9.98|10.61|.77|.17|
|Capsule Tail|.250|.250|.250|1.000|.250|0|.00000|.750|.03|.04|.06|0|
|Heuristic Hybrid|1.000|1.000|1.000|1.000|.473|8,192|.00061|.000|10.06|10.59|.77|.17|
|Literal Search|1.000|1.000|1.000|1.000|.473|8,192|.00061|.000|7.17|7.71|.75|.09|
|Oracle|1.000|1.000|1.000|1.000|.723|74|.75000|.000|.04|.06|.08|0|
|Random|.792|1.000|1.000|1.000|.264|50,015|.00061|.000|2.03|2.53|.17|.09|

### LogDx-CI 35

|方法|R64|R128|R256|Supporting|nDCG|First B|Density|Miss|P50 ms|P95 ms|RSSΔ P95 MiB|扫描 MiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|BM25|.749|.824|.893|.920|.846|1,334|.01491|.057|34.31|344.53|1.64|2.11|
|Capsule Tail|.703|.703|.703|.723|.759|0|.00000|.086|.03|.05|.06|0|
|Heuristic Hybrid|.780|.853|.893|.920|.851|5,305|.01481|.057|35.91|348.06|1.75|2.11|
|Literal Search|.803|.864|.893|.914|.861|0|.01485|.086|29.70|339.33|6.89|1.06|
|Oracle|1.000|1.000|1.000|1.000|.976|13|.99825|.000|.07|.09|.08|0|
|Random|.731|.843|.873|.908|.841|5,542|.01414|.029|2.36|6.20|6.75|1.06|

First B 是成功命中 Case 的条件均值，必须和 Miss 一起阅读。完整 JSON 还保存每项指标的
95% CI、相对 Capsule Tail 的逐 Case 配对差值和 split 分解。LogDx-CI 多数日志较短，Random
在256 KiB预算下也达到.873，不能只依赖R256判断算法优劣。

### Scale Stress 9

除 Oracle 外，五个基线在全部前/中/尾位置均未召回人工埋入的稀有 anchor。这不是评分器
错误：BGL/Spark 中高频 `FATAL/failed` 块压过了真正稀有证据，暴露出现有 query/异常模式
策略缺少全局稀有度。压力集总体 P95 中，Random 约656 ms但增量RSS约696 MiB；Literal
约28.6 s且约701 MiB；BM25/Hybrid约42.1/42.0 s、最大 Case 双遍扫描1,000 MiB，增量RSS低于2 MiB。
该集合只证明检索规模与资源行为，不证明 Data/ML 根因诊断正确率。

Defects4ML v1 的 `bugs.zip` 已按官方 MD5 与 SHA-256 校验，并在四个固定 x86 TensorFlow
容器中重放15个低资源候选，并预置经校验的 MNIST/CIFAR-10 缓存。076、093、082、087、079、
080共6个满足 buggy 非零且 fixed 为零；081、090、054、053的 fixed 在600秒内超时，另外5个
fixed 仍失败，所以没有发布历史 bundle，也没有用合成 Case 补位。候选原始
日志和返回码保存在临时重放目录，机器可读摘要为
`benchmark-results/v1/historical-replay-audit.json`。

正式验收仍被三项事实门禁阻止：12个公开历史故障仅有6个候选完成双重放；12个最终 Core test
尚未完成双人Gold复核；当前环境没有模型凭据，因此141次Agent策略运行与人工诊断评分尚未
开始。机器可读结果位于 `benchmark-results/v1/retrieval-comparison.json`，不得把当前92
Case结果描述成正式完整60-case Core或Agent排行榜。

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

旧基线的工作区验收结果：

```text
Ruff: passed
mypy: passed
WIRE_PROTOCOL.md: up to date
unit: 425 passed
integration: 18 passed
pytest: 443 passed in 40.44s
```

Workflow Contract 增量的当前自动化验收结果：

```text
Ruff: passed
mypy: passed
WIRE_PROTOCOL.md v2: up to date
unit: 455 passed（含 14 个 Evidence Retrieval Benchmark 测试）
integration: 18 passed
pytest: 473 passed in 67.67s
VS Code typecheck/unit: passed, 8 tests
VS Code Extension Host: passed
VSIX package: passed
```

这些结果验证实现和协议闭环，不替代尚未执行的三类真实 workload Pilot。

## VS Code Alpha 验收

2026-07-29 增加的薄型 VS Code 客户端未修改 Incident Agent、进程监督或补丁应用策略，
因此没有重跑上述 9-case LLM Pilot。新增验收覆盖：

- TypeScript 编译与 8 个纯 UI、cursor、Trust 和上下文动作测试；
- 真实 VS Code Extension Host 启动、daemon 连接、命令注册和 snapshot refresh；
- 真实 daemon、真实训练子进程的 `launch.preview → launch.start → job.read_log`；
- VSIX 成功打包；关闭扩展只断开客户端；
- 5 组交替运行中，无日志读者中位 0.8126 秒，300 ms cursor 轮询中位 0.8076 秒，
  观测差值 -0.61%，未发现可测的训练开销。

性能数字来自 0.75 秒 sleep fixture，只用于排除轮询造成的明显固定开销，不外推到真实模型
吞吐或 GPU 利用率。
