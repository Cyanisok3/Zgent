export const WIRE_PROTOCOL_VERSION = 2;

export type JsonObject = Record<string, unknown>;

export interface CoreStartResult {
  status: "started" | "already_running";
  host: string;
  port: number;
  pid?: number | null;
  workspace_root: string;
  protocol_version: number;
}

export interface PongResult {
  server_version: string;
  protocol_version: number;
  startup_workspace_root: string;
  uptime_ms: number;
  received_at: string;
}

export interface LaunchPreview {
  argv: string[];
  cwd: string;
  env_overrides: Record<string, string>;
  executable: string;
  config_paths: string[];
  fingerprint: string;
}

export interface JobSnapshot {
  job: {
    id: string;
    status: string;
    created_at: string;
    updated_at: string;
    current_attempt_id?: string | null;
  };
  argv: string[];
  workspace_root: string;
  attempt?: {
    id: string;
    status: string;
    started_at: string;
    finished_at?: string | null;
    returncode?: number | null;
    signal?: number | null;
  } | null;
  incident?: {
    id: string;
    status: string;
    active_proposal_id?: string | null;
    last_outcome?: string | null;
  } | null;
  diagnosis?: {
    id: string;
    category: string;
    summary: string;
    root_cause: string;
    confidence: number;
    causal_support: "direct" | "inferred";
    patch_recommended: boolean;
    evidence: EvidenceRef[];
  } | null;
  proposal?: {
    id: string;
    files: Array<{ path: string; change_type: string }>;
  } | null;
  smoke_config?: JsonObject | null;
  smoke_config_fingerprint?: string | null;
  can_apply?: boolean;
}

export interface EvidenceRef {
  source: "stdout" | "stderr" | "workspace";
  reference: string;
  description: string;
}

export interface IncidentReview {
  proposal_id: string;
  path: string;
  before_text: string;
  after_text: string;
}

export interface LogChunk {
  data: string;
  next_offset: number;
  total_bytes: number;
  eof: boolean;
}

export interface StatusPresentation {
  text: string;
  tooltip: string;
  icon: string;
}

// 根据诊断支持强度和 Proposal 状态生成确定性的修复处置说明
export function diagnosisDisposition(snapshot: JobSnapshot): string {
  const diagnosis = snapshot.diagnosis;
  if (diagnosis === null || diagnosis === undefined) {
    return "Diagnosis is not available yet.";
  }
  if (diagnosis.causal_support === "inferred") {
    return "Automatic patch blocked because the causal support is inferred. Provide additional evidence via /incident.";
  }
  if (!diagnosis.patch_recommended) {
    return "The cause is evidence-backed but outside cyan's safe single-file repair boundary.";
  }
  if (snapshot.proposal !== null && snapshot.proposal !== undefined) {
    return "An evidence-backed single-file proposal is ready for approval.";
  }
  if (snapshot.incident?.status === "diagnosing") {
    return "cyan is validating an evidence-backed single-file proposal.";
  }
  return "No valid proposal was created; the diagnosis was retained.";
}

const ACTIVE_JOBS = new Set(["starting", "running"]);

// 判断快照中的真实训练进程是否仍在运行
export function isActiveJob(snapshot: JobSnapshot | undefined): boolean {
  return snapshot !== undefined && ACTIVE_JOBS.has(snapshot.job.status);
}

// 将 daemon 快照映射为稳定的状态栏文案
export function statusPresentation(
  connected: boolean,
  snapshot: JobSnapshot | undefined,
): StatusPresentation {
  if (!connected) {
    return { text: "Offline", tooltip: "cyan core is not connected", icon: "debug-disconnect" };
  }
  if (snapshot === undefined) {
    return { text: "Idle", tooltip: "No attached training job", icon: "circle-outline" };
  }
  const incident = snapshot.incident?.status;
  if (incident === "diagnosing") {
    return { text: "Investigating", tooltip: "cyan is diagnosing a training failure", icon: "sync~spin" };
  }
  if (incident === "awaiting_approval") {
    return { text: "Action required", tooltip: "A proposed fix is ready for review", icon: "warning" };
  }
  if (incident === "resolved") {
    return { text: "Resolved", tooltip: "The original training command succeeded", icon: "check" };
  }
  if (incident === "unresolved" || incident === "rollback_blocked" || incident === "stale") {
    return { text: "Unresolved", tooltip: `Incident status: ${incident}`, icon: "error" };
  }
  if (isActiveJob(snapshot)) {
    return { text: "Training", tooltip: "cyan is supervising the training process", icon: "sync~spin" };
  }
  return { text: snapshot.job.status, tooltip: `Job status: ${snapshot.job.status}`, icon: "circle-filled" };
}

// 把结构化 diagnosis 渲染为只读 Markdown 详情
export function diagnosisMarkdown(snapshot: JobSnapshot): string {
  const diagnosis = snapshot.diagnosis;
  if (diagnosis === null || diagnosis === undefined) {
    const outcome = snapshot.incident?.last_outcome;
    const note = outcome === "smoke_evidence_unavailable"
      ? "Smoke failed and the patch was rolled back, but no Smoke output was available; automatic re-diagnosis was skipped."
      : "Diagnosis is not available yet.";
    return `# cyan diagnosis\n\n${note}\n`;
  }
  const evidence = diagnosis.evidence
    .map((item) => `- **${item.source}** — ${item.description}\n  - \`${item.reference}\``)
    .join("\n");
  return [
    "# cyan diagnosis",
    "",
    `**Category:** ${diagnosis.category}`,
    `**Causal support:** ${diagnosis.causal_support}`,
    `**Patch recommended:** ${diagnosis.patch_recommended ? "yes" : "no"}`,
    `**Repair decision:** ${diagnosisDisposition(snapshot)}`,
    "",
    "## Summary",
    "",
    diagnosis.summary,
    "",
    "## Root cause",
    "",
    diagnosis.root_cause,
    "",
    "## Evidence",
    "",
    evidence || "No evidence is available.",
    "",
  ].join("\n");
}
