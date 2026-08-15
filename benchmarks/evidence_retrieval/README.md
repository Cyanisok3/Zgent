# Cyan Evidence Retrieval Benchmark v1

该 Benchmark 独立于生产 Incident Runtime，用固定 Failure Capsule、封存日志和 byte-range gold
比较证据检索方法。它不训练 GP，也不会把 Skill 或 Subagent 注册到真实 Incident session。

## 安装与数据层级

真实 Core workload 的依赖独立于 Cyan 生产运行时：

```bash
uv sync --group benchmark
```

- `cyan_core`：12 个 Pandas、scikit-learn、PyTorch CPU adapter 各执行四类真实故障，共48个
  失败/修复双重放 Case；再加12个公开历史故障。缺少历史故障或 test 双人 Gold 复核时，
  complete-core 校验会失败。
- `external_generalization`：固定 LogDx-CI v1.2 的 35 个公开 CI Case，仅作外部测试。
- `scale_stress`：用户提供 Hadoop、Spark、BGL 的 LogHub 原始日志，生成 5/50/500 MiB 的
  九个规模测试。结果只代表检索规模，不代表 ML 诊断正确性。

原始大日志和运行结果不进入 Git。Manifest 保存来源、许可、日志 SHA-256、稳定 byte range、
重放凭证和标注审核次数。

`profile=ci` 使用8个无 Data/ML 依赖的小型子进程 fixture，只验证 schema、预算、offset 和
确定性；它们不进入正式 Core 排行榜。

## 快速验证

```bash
uv run python scripts/benchmark_evidence.py prepare \
  --corpus /private/tmp/cyan-benchmark-ci --profile ci

uv run python scripts/benchmark_evidence.py run \
  --corpus /private/tmp/cyan-benchmark-ci \
  --method heuristic_hybrid \
  --output benchmark-results/hybrid.run.json

uv run python scripts/benchmark_evidence.py score \
  --corpus /private/tmp/cyan-benchmark-ci \
  --run benchmark-results/hybrid.run.json \
  --output benchmark-results/hybrid.scores.json

uv run python scripts/benchmark_evidence.py compare \
  --scores benchmark-results/*.scores.json \
  --output benchmark-results/retrieval-comparison.json
```

首版检索器为 `random`、`capsule_tail`、`literal_search_policy`、`bm25`、
`heuristic_hybrid` 和只作上限校验的 `oracle`。每个方法共享 32 KiB Capsule、256 KiB
额外证据预算和 32 KiB 单块上限。

## 完整语料

生成 48 个受控 Case；未提供 12 个历史 bundle 时命令会以非零状态指出语料未完成：

```bash
uv run python scripts/benchmark_evidence.py prepare \
  --corpus /path/to/corpus --profile core \
  --historical-source /path/to/reviewed-historical-bundles
```

历史 bundle 必须使用同一 `CaseManifest`，包含公开 commit 对或 issue + revision 对、真实失败
和成功 return code、日志摘要。`--allow-incomplete-core` 仅供开发检索器使用，不能用于正式结果。

Defects4ML 候选可使用仓库内四个固定基础镜像的 Dockerfile 重放；命令会断网执行、保存
每个候选的原始日志，并且只有达到12个“buggy 非零、fixed 为零”的 Case 才发布 bundle：

```bash
uv run python scripts/benchmark_evidence.py history-prepare \
  --archive /path/to/bugs.zip \
  --output /path/to/historical-bundles \
  --work-dir /path/to/replay-work
```

2026-08-15 对15个低资源候选的实跑仅有076、093、082、087、079、080六个合格，因此发布门禁
按设计失败；完整审计位于 `benchmark-results/v1/historical-replay-audit.json`。没有用合成
Case补足数量。

Core test Gold 先导出两份独立模板；只有 reviewer ID 不同且内容完全一致时才写回批准状态：

```bash
uv run python scripts/benchmark_evidence.py gold-review-export \
  --corpus /path/to/corpus --output reviewer-a.json

uv run python scripts/benchmark_evidence.py gold-review-import \
  --corpus /path/to/corpus --review reviewer-a.json reviewer-b.json
```

导入固定 LogDx-CI v1.2；省略 `--logdx-source` 时会下载并校验固定 archive SHA-256：

```bash
uv run python scripts/benchmark_evidence.py prepare \
  --corpus /path/to/corpus --profile external \
  --cache-dir /path/to/cache
```

压力集要求显式提供三个已下载的 LogHub 文件：

```bash
uv run python scripts/benchmark_evidence.py prepare \
  --corpus /path/to/corpus --profile stress \
  --loghub-source hadoop=/data/Hadoop.log \
  --loghub-source spark=/data/Spark.log \
  --loghub-source bgl=/data/BGL.log
```

## GP 特征与 Agent 策略

```bash
uv run python scripts/benchmark_evidence.py export-features \
  --corpus /path/to/corpus --output benchmark-results/candidates.jsonl

uv run python scripts/benchmark_evidence.py agent-run \
  --corpus /path/to/corpus --strategy retrieval_skill \
  --model claude-sonnet-4-6 \
  --runs-dir benchmark-results/runs \
  --output benchmark-results/skill.json
```

特征导出默认排除 test。Agent Track 只接受 `cyan_core/test` 和 LogDx 外部 Case，可选
`current_agent`、`retrieval_skill`、`readonly_subagents`。Subagent 是 Benchmark 专用的三个
顺序只读角色，共享证据和 token 预算，不使用生产通用 `spawn_agent`。正式运行默认要求
完整60-case Core、12个已双审 test Case 和35个 LogDx-CI，共47个 Case；
`--allow-incomplete-core` 仅用于本地调试。

三种策略运行完后，导出隐藏 Case/策略的随机盲审包，再合并两名复核者的 CSV：

```bash
uv run python scripts/benchmark_evidence.py review-export \
  --corpus /path/to/corpus --agent-results benchmark-results/*agent.json \
  --pack review-pack.json --key private-review-key.json \
  --template review.csv --seed 0

uv run python scripts/benchmark_evidence.py agent-score \
  --agent-results benchmark-results/*agent.json --key private-review-key.json \
  --review reviewer-a.csv reviewer-b.csv --output agent-scores.json

uv run python scripts/benchmark_evidence.py agent-compare \
  --scores agent-scores.json --output agent-comparison.json
```

`compare` 始终按 tier 分层，并将 `combined_score` 固定为 `null`，防止将受控、外部和压力
数据混成一个误导性分数。完整检索结果还包含所有指标的 bootstrap 95% CI、相对 Capsule
Tail 的配对差值，以及 Scale Stress 的逐规模/证据位置明细。

## 当前运行状态

2026-08-15 已在临时语料上完成48个真实受控 Core、35个 LogDx-CI 和9个完整
5/50/500 MiB Scale Stress Case，并运行六个检索器。机器可读结果位于
`benchmark-results/v1/retrieval-comparison.json`（该目录默认不提交 Git）。
公开数据校验和、Python包版本和历史重放镜像分别冻结在 `sources.lock.json` 与
`environment.lock.json`。

正式60-case Core 与 Agent 排行榜尚受三个门禁约束：12个历史故障重放、12个 Core test
双人审核、以及本机模型凭据。门禁未完成前不得将当前结果描述为正式完整验收。
