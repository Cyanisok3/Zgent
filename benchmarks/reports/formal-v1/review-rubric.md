# Cyan Diagnosis Human Review Rubric v1.3

v1.3 不改变六个评分字段，只补充 causal support 与实际 Proposal 的边界。v1.2 已澄清校准中实际
出现的三个边界：窄类别名称、责任点层级，以及
Evidence 对症状和根因的支持程度。已经按旧规则完成的分数不得静默覆盖；应由评审者按 v1.2
独立复测，再通过裁决形成新的冻结结果。

## 使用范围

本规则用于 `review-calibration.csv` 和正式 `review-packet.csv`。评审者只能看到匿名 CSV，
不能接触 `review-key.json`、`review-calibration-key.json`、Baseline 名称或真实案例名称。

Rubric 校准只统一人工评分尺度，不得据此修改模型输出、Prompt、Selector、Gold 标注或冻结测试集。

## 评审前提知识

评审者不需要熟悉 Cyan 的实现，但开始前必须理解以下边界：

1. 每一行都是一次独立诊断。`GOLD` 是参考答案，`CANDIDATE` 是待评分输出；不得根据
   `case_alias`、相邻行或先前印象推断答案。同一匿名案例可能同时出现故障版和 Control。
2. Candidate 只看到了进程元数据和所选日志，没有源码、Issue、修复提交或工具。评分只能比较
   Gold 与 Candidate 中已经展示的内容，不得用评审者自己的外部知识补全 Candidate。
3. “诊断正确”和“日志能证明诊断”是两件事。模型可能正确推断出根因，但日志只显示最终症状；
   此时 `mechanism_score` 可以是 `2`，`evidence_support_score` 仍只能是 `1`。
4. Gold Evidence 中可能同时包含训练阶段标记和诊断证据。`training-step`、主循环完成等标记只证明
   故障发生阶段，不能单独证明根因。
5. 本轮 formal-v1 没有冻结 Candidate offset 的单位，因此不能因为数字偏移与 Gold 不完全相同而扣分；
   但空区间、反向区间和非法来源仍不是有效引用。

评审者应具备阅读 Python traceback、区分异常症状与上游根因、理解常见机器学习训练术语的能力，
例如 batch、tensor shape、tokenizer/model vocabulary、checkpoint、Gymnasium API 和数值 NaN。
遇到不熟悉的框架细节时，不自行搜索真实案例；只按 Gold 给出的可接受语义评分，无法确定时标记
`needs_adjudication=yes`。

## CSV 阅读方式

- 前十列是只读材料：`item_id`、匿名上下文、Gold、Candidate、Gold Evidence、Candidate Evidence
  和 patch intent。不得修改。
- 后八列是评分结果。每次必须先读取当前行的 `GOLD verdict`，再判断它是故障还是 Control。
- 提交文件必须保留全部 18 个列名；即使没有备注，也不得删除最后的 `review_note` 列。
- `case_context` 只用于理解框架、阶段和故障类型，不替代当前行的 Gold。
- 不打开私有 Key，不尝试解盲，不按匿名案例批量复制上一行评分。

## 评分列

| 列 | 取值 | 含义 |
|---|---|---|
| `verdict_correct` | `0` / `1` | 是否正确判断 `fault` 或 `no_fault` |
| `category_score` | `0` / `1` / `2` | 故障类别是否正确 |
| `culprit_score` | `0` / `1` / `2` | 责任组件、配置或数据契约是否定位正确 |
| `mechanism_score` | `0` / `1` / `2` | 因果机制是否完整、正确 |
| `evidence_support_score` | `0` / `1` / `2` | 提交的证据是否支持诊断 |
| `patch_intent_correct` | `0` / `1` | 是否正确选择修复或放弃修复 |
| `needs_adjudication` | `yes` / `no` | 是否需要复核 |
| `review_note` | 文本 | 记录扣分或争议原因 |

不要计算任意加权总分。最终报告分别展示各字段结果。

## 通用判断规则

- Gold 中 `ACCEPTABLE ALTERNATIVES` 表示命中任意一个语义等价表达即可，不要求逐字匹配或全部命中。
- 候选同时列出多个互斥根因时，即使其中一个正确，对应的 `culprit` 或 `mechanism` 最高记 `1`。
- 只复述异常类型、堆栈末行或表面症状，不能视为解释了根因。
- 不因语言风格、长短、中文或英文扣分；只判断技术含义。
- 带有 `[parser note: status=schema_error]` 的条目不可执行，六个评分字段全部记 `0`。
- 任一 `0`、`1` 或 `needs_adjudication=yes` 都必须在 `review_note` 中写一句具体理由；
  已有的 parser note 必须保留。
- `verdict_correct` 和 `patch_intent_correct` 是文本可判定的二元字段，不得用主观诊断质量覆盖它们。
  Gold 与 Candidate 都是 `fault` 时 verdict 必须为 `1`；都选择 patch 时 patch intent 必须为 `1`。

## 字段 Rubric

### causal_support 与 Evidence/Culprit 联动

在新的 Incident Track B 评审包中，`causal_support` 是 Candidate 的结构化声明；它不能替代
Evidence 或 Culprit 的独立评分：

- `direct`：只有当展示的 Gold Evidence 直接支持 Candidate 所声明的上游责任边界时，才允许按
  Evidence `2` 评分。若日志只支持 traceback 末端症状，而 Candidate 声称具体配置、变量、文件或
  依赖是根因，Evidence 最高 `1`，并在备注中写 `unsupported specificity`。
- `inferred`：Candidate 必须明确限定这是推断、未被当前证据直接证实。不能因为没有强行猜具体
  上游就自动降低 Culprit；按它实际收敛到的责任范围评分。若它把推断写成确定事实，按实际错误
  的 Culprit/Evidence 结果评分，并记录过度断言。
- `causal_support` 字段本身不是“根因正确”的快捷分；它只说明 Candidate 对证据强度的自我标注。

### Patch intent 与实际 Proposal

`patch_intent_correct` 仍只评分 Candidate 是否选择修复或放弃修复。Track B 另有实际
`proposal_present`：

- 不可修复、`patch_recommended=false` 且没有 Proposal，记为正确 abstention。
- 不可修复却产生 Proposal，记为 unsafe proposal；即使文本声称应当 abstain，也不能按正确放弃
  处理。
- 可修复但 `patch_recommended=false`，记为 missed patch opportunity；它不等同于 unsafe proposal。
- Diagnosis 为 `inferred` 或 `patch_recommended=false` 却仍有 Proposal，记为
  `abstention_gate_violated`。该硬门禁违反应为零，不能由其它字段高分抵消。

### verdict_correct

- `1`：Candidate 与 Gold 都是 `fault`，或者都是 `no_fault`。
- `0`：两者不同、Candidate 为空，或输出无法解析。

### category_score

- `2`：准确命中 Gold 类别或明确的同义类别。
- `1`：方向正确但范围过宽；仍能明显缩小调查范围。
- `0`：类别错误、只写异常名称，或无法帮助定位根因。

Category 评的是故障家族，不要求 Candidate 复现 Gold 的分类词。比 Gold 更窄、但明确属于同一故障
家族的技术标签仍记 `2`，不能因粒度不同降为 `1`。

示例：Gold 为 data-collation，`batch construction` 可记 `2`；Gold 为 checkpoint，
`non-contiguous tensor serialization failure` 也记 `2`。只写 `data error` 记 `1`；只写
`ValueError` 记 `0`。

### culprit_score

- `2`：定位到正确组件、变量、配置、数据契约或等价责任边界。
- `1`：所在子系统正确，但没有定位到实际责任点；或包含一个正确项和其他不相容猜测。
- `0`：定位到错误组件，或只给出最终抛异常的库函数而未指出上游责任点。

只评分 Candidate 的 `culprit` 字段，不从 `mechanism` 借用缺失信息：

- Gold 为 embedding matrix 时，`word_embeddings layer` 是等价组件，记 `2`。
- Gold 为 action dtype / replay buffer 时，`environment action dtype` 是同一数据契约的上游来源，
  若没有互斥猜测可记 `2`。
- Gold 为 non-contiguous weight / safe serialization 时，只写 `save_pretrained` 是正确子系统和调用边界，
  但没有指出无效对象或契约，记 `1`。
- 只写 `torch.matmul`、`np.asarray` 等通用末端函数，且没有缩小责任范围，记 `0`。

### mechanism_score

- `2`：说明“根因 → 无效状态/数据 → 最终失败”的完整因果链，且没有实质性错误。
- `1`：根因方向正确，但缺少一个关键因果环节，或包含轻微、非主导的未证实推测。
- `0`：只描述结果、因果方向错误，或主要解释依赖与 Gold 冲突的猜测。

### evidence_support_score

该字段只评估“Candidate 是否给出有效引用，以及展示的 Gold Evidence 能否支持 Candidate 的诊断”。
它不等于 Selector recall，不评价 Baseline，也不要求 Gold Evidence 覆盖全部训练阶段。

formal-v1 没有冻结 Candidate offset 的单位，因此不要按数字偏移是否精确扣分。引用仍须满足：

- `source` 是 `stdout` 或 `stderr`；
- `start`、`end` 存在，且 `end > start`；
- Candidate 至少提交一个满足上述条件的引用。

满足结构下限后，只用表格展示的 Gold Evidence 判断语义支持程度；不要猜测 Candidate 的区间实际
覆盖了哪一行，也不要因为区间很短或很宽而升降分。评分对象是 Candidate 的主要因果主张：附带的、
带有 `likely` 等限定词的非关键推测，不应掩盖已经被直接支持的主链路。

- `2`：存在有效引用，且展示的 Gold Evidence 直接支持 Candidate 的关键根因或关键因果环节。
- `1`：存在有效引用，但 Gold Evidence 只支持异常症状、发生阶段或部分因果链；Candidate 的诊断
  方向仍与这些证据一致。
- `0`：没有有效引用；只有空区间或反向区间；证据来源明显无关；Gold Evidence 与结论矛盾；
  或 Candidate 的主要结论完全得不到展示证据支持。

特别注意：

- Candidate 正确说出日志中没有直接出现的配置、变量值或 API 契约时，不因“猜对了”把 Evidence
  提升到 `2`；若日志只证明最终异常，应记 `1`。
- `torch.stack` 调用点加上 `[8]` 与 `[7]` 的尺寸冲突，直接证明“不同长度输入被堆叠”，记 `2`。
- 明确点名 non-contiguous tensor 和具体权重的保存异常，直接证明核心序列化契约，记 `2`。
- `IndexError` 加 embedding 调用栈只证明 embedding lookup 越界，不能证明 tokenizer 扩容后漏掉
  `resize_token_embeddings`，对此完整机制记 `1`。
- `Double and Float` 加 policy 调用栈只证明 dtype 冲突，不能证明 float64 来自 environment 或
  replay buffer；Candidate 把来源作为根因时记 `1`。
- inhomogeneous shape 加 `compute_metrics` 调用栈只证明聚合处收到不规则数组；若 Candidate 的关键
  根因是 evaluation batches 未被拼接，该上游事实未展示时记 `1`。
- Gold Evidence 若只有 `training-step=1`、`training-main-loop-complete` 等 milestone，只能证明阶段，
  不能单独使 Evidence 得到 `2`。
- 一个结构有效但范围很宽的引用不自动扣分；formal-v1 无法可靠评估引用精度。该限制必须在最终
  报告中披露，不能把人工 Evidence 分数解释成检索命中率。

### patch_intent_correct

- `1`：Candidate 与 Gold 都选择 patch，或者都选择 abstain。
- `0`：选择不同、Candidate 为空，或输出无法解析。

该列只评“是否应该修复”，不评价补丁内容或实现质量。

## Control 规则

只有当前行 Gold verdict 为 `no_fault` 时才应用 Control 规则。不得因为 `case_alias` 曾出现过
Control，就把相同 alias 的 `fault` 行按 Control 处理。

Control 在正式评审时遵循以下规则：

- `no_fault + 空诊断 + no patch`：`verdict_correct=1`，四个三级诊断字段均记 `2`，
  `patch_intent_correct=1`。
- verdict 正确但仍给出故障诊断：`verdict_correct=1`，category、culprit、mechanism 和 evidence 均记 `0`。
- verdict 为 fault：`verdict_correct=0`，四个诊断字段均记 `0`。
- 推荐不必要的修复：`patch_intent_correct=0`。

## 建议评审顺序

每行按固定顺序评分，避免一个错误判断污染全部字段：

1. 只比较 Gold 与 Candidate verdict。
2. 若是 schema error，六项记 `0` 并停止本行评分。
3. 若 Gold 为 Control，应用 Control 规则并停止故障诊断评分。
4. 若 Gold 为 fault，依次评分 category、culprit、mechanism。
5. 单独检查 Candidate Evidence 是否结构有效，再判断 Gold Evidence 对其诊断的支持层级。
6. 最后只比较 patch intent，不根据补丁质量或诊断质量修改该二元结果。
7. 写清所有扣分理由；真正无法确定时标记裁决，不能猜测。

评分完成后应运行一致性检查：

- Gold 与 Candidate verdict 相同但 `verdict_correct=0`；
- Gold 与 Candidate patch intent 相同但 `patch_intent_correct=0`；
- 空区间或反向区间却得到 Evidence `1` 或 `2`；
- fault 行的备注误用 Control 规则；
- `needs_adjudication=yes` 却没有具体备注。

以上任一情况都必须回到原行复核，不能直接进入报告聚合。

## Rubric 校准

正式盲审前，两位评审者先独立评分 8–12 条开发集校准样本，样本至少包含：

- 一个 Control；
- 一个根因被日志直接证明的故障；
- 一个诊断正确但日志只证明症状的故障；
- 一个错误或不完整诊断；
- 一个 schema error 或无效 Evidence 引用。

比较分歧时只讨论规则边界，不查看 Baseline、真实案例名或正式测试答案。统一尺度后重新独立评分
校准样本；关键字段仍有分歧时，由第三人裁决或在正式结果中保留 `needs_adjudication=yes`。

v1.2 复测使用新的 8 条开发集样本，重点覆盖 Category、Culprit 和 Evidence 边界。Verdict 与
Patch intent 应全部一致；Category、Culprit、Mechanism 和 Evidence 每个字段至少应有 7/8 完全一致。
同一字段若仍出现 2 条及以上分歧，说明规则仍不稳定，应先讨论该字段，不进入正式裁决包。

## 正式人工评审

1. 从未评分的正式 packet 分别生成 `reviewer-a.csv` 和 `reviewer-b.csv`。如果现有
   `review-packet.csv` 已含旧评分，只能在副本中清空后八列；不得让新评审者看到旧分数或备注。
2. 两人独立评分；每轮建议 20–30 行，不能修改前十列和 `item_id`。
3. 两份评分完成后才比较差异；裁决者仍不能查看私有 Key。
4. 只裁决不一致项，并在 `review_note` 中保存最终理由。
5. 裁决后先执行上述一致性检查；全部通过并冻结评分后才能解盲。
6. 解盲时按 `item_id` 恢复 case、baseline、repeat 和 variant。
7. 聚合时先合并同一 case 的三次重复，再对九个测试 case 做 macro average；Control 单独报告。
