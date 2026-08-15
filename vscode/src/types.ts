export const WIRE_PROTOCOL_VERSION = 2;

export type JsonObject = Record<string, unknown>;

export interface CoreStartResult {
  status: "started" | "already_running";
  host: string;
  port: number;
  pid?: number | null;
  workspace_root: string;
  protocol_version?: number | null;
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
  workflow?: WorkflowSummary | null;
}

export interface WorkflowSummary {
  version: number;
  artifact_count: number;
  check_count: number;
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
  workflow?: WorkflowSummary | null;
  attempt?: {
    id: string;
    status: string;
    started_at: string;
    finished_at?: string | null;
    returncode?: number | null;
    signal?: number | null;
    phase?: "preflight" | "main" | "postflight";
    check_id?: string | null;
  } | null;
  incident?: {
    id: string;
    status: string;
    active_proposal_id?: string | null;
  } | null;
  diagnosis?: {
    id: string;
    category: string;
    summary: string;
    root_cause: string;
    confidence: number;
    evidence: EvidenceRef[];
    recovery?: Recovery | null;
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
  source: "stdout" | "stderr" | "workspace" | "contract";
  reference: string;
  description: string;
}

export interface Recovery {
  kind: "patch" | "operator_action" | "none";
  summary: string;
  actions: Array<{
    target?: string | null;
    instruction: string;
    verification: string;
  }>;
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
  if (incident === "action_required") {
    return { text: "Action required", tooltip: "Complete the operator actions, then recheck the workflow", icon: "warning" };
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
    return "# cyan diagnosis\n\nDiagnosis is not available yet.\n";
  }
  const evidence = diagnosis.evidence
    .map((item) => `- **${item.source}** — ${item.description}\n  - \`${item.reference}\``)
    .join("\n");
  const recovery = diagnosis.recovery === null || diagnosis.recovery === undefined
    ? "Legacy diagnosis: no structured recovery is available."
    : [
      `**Kind:** ${diagnosis.recovery.kind}`,
      diagnosis.recovery.summary,
      ...diagnosis.recovery.actions.map((action) => (
        `- ${action.target === null || action.target === undefined ? "" : `**${action.target}:** `}`
        + `${action.instruction}\n  - Verify: ${action.verification}`
      )),
    ].join("\n\n");
  return [
    "# cyan diagnosis",
    "",
    `**Category:** ${diagnosis.category}`,
    `**Confidence:** ${Math.round(diagnosis.confidence * 100)}%`,
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
    "## Recovery",
    "",
    recovery,
    "",
  ].join("\n");
}
