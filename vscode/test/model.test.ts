import * as assert from "node:assert/strict";
import { test } from "node:test";
import { initialTailOffset, terminalText } from "../src/terminalSupport";
import {
  JobSnapshot,
  diagnosisDisposition,
  diagnosisMarkdown,
  isActiveJob,
  statusPresentation,
} from "../src/types";

// 构造最小 Job 快照供纯状态映射测试复用
function snapshot(jobStatus: string, incidentStatus?: string): JobSnapshot {
  return {
    job: {
      id: "job-1",
      status: jobStatus,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      current_attempt_id: "attempt-1",
    },
    argv: ["python", "train.py"],
    workspace_root: "/tmp/project",
    attempt: {
      id: "attempt-1",
      status: jobStatus,
      started_at: "2026-01-01T00:00:00Z",
    },
    incident: incidentStatus === undefined
      ? null
      : { id: "incident-1", status: incidentStatus },
    diagnosis: null,
    proposal: null,
  };
}

// 功能：验证 running/starting 才允许显示训练期 Cancel 动作
// 设计：测试纯状态谓词，避免依赖 VS Code Tree View 实例
test("isActiveJob only accepts real running states", () => {
  assert.equal(isActiveJob(snapshot("starting")), true);
  assert.equal(isActiveJob(snapshot("running")), true);
  assert.equal(isActiveJob(snapshot("failed")), false);
});

// 功能：验证 Incident 状态优先于 Job 状态映射到状态栏
// 设计：以 awaiting_approval 覆盖 failed Job，锁定 Action required 文案
test("statusPresentation prioritizes actionable incident", () => {
  const status = statusPresentation(true, snapshot("failed", "awaiting_approval"));
  assert.equal(status.text, "Action required");
  assert.equal(status.icon, "warning");
});

// 功能：验证长日志恢复只请求有界尾部且终端换行稳定
// 设计：直接测试 cursor 和文本纯函数，避免伪造 daemon 日志结果
test("terminal helpers bound the initial replay", () => {
  assert.equal(initialTailOffset(100_000, 32_768), 67_232);
  assert.equal(initialTailOffset(10, 32_768), 0);
  assert.equal(terminalText("one\ntwo\r\n"), "one\r\ntwo\r\n");
});

// 功能：验证 diagnosis Markdown 包含根因、置信度和 evidence 引用
// 设计：填入一个真实形状的 Diagnosis 结构并检查可读输出
test("diagnosisMarkdown renders evidence", () => {
  const value = snapshot("failed", "awaiting_approval");
  value.diagnosis = {
    id: "diagnosis-1",
    category: "device",
    summary: "checkpoint load failed",
    root_cause: "CUDA checkpoint on a CPU host",
    confidence: 0.9,
    causal_support: "direct",
    patch_recommended: false,
    evidence: [
      {
        source: "stderr",
        reference: "stderr:0-120",
        description: "torch.load raised CUDA error",
      },
    ],
  };
  const rendered = diagnosisMarkdown(value);
  assert.match(rendered, /CUDA checkpoint/);
  assert.match(rendered, /Repair decision:/);
  assert.match(rendered, /direct/);
  assert.match(rendered, /stderr:0-120/);
  assert.equal(
    diagnosisDisposition(value),
    "The cause is evidence-backed but outside cyan's safe single-file repair boundary.",
  );
});

// 功能：验证 VS Code 对推断根因显示安全的 abstention 说明
// 设计：不提供 Proposal，确保 UI 不把模型置信度渲染成可执行建议
test("diagnosisDisposition explains inferred repair abstention", () => {
  const value = snapshot("failed", "unresolved");
  value.diagnosis = {
    id: "diagnosis-2",
    category: "runtime",
    summary: "runtime failure",
    root_cause: "Inference — not directly established by observed evidence: producer mismatch",
    confidence: 0.99,
    causal_support: "inferred",
    patch_recommended: false,
    evidence: [],
  };
  assert.match(diagnosisDisposition(value), /inferred/);
});
