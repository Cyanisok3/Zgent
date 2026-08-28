import * as path from "node:path";
import * as vscode from "vscode";
import { JobSnapshot, diagnosisDisposition, isActiveJob } from "./types";

type NodeKind =
  | "action"
  | "job"
  | "detail"
  | "incident"
  | "diagnosis"
  | "evidence"
  | "proposal";

export interface CyanNode {
  kind: NodeKind;
  label: string;
  description?: string;
  tooltip?: string;
  contextValue?: string;
  command?: vscode.Command;
  children?: CyanNode[];
  icon?: string;
}

export class CyanTreeProvider implements vscode.TreeDataProvider<CyanNode> {
  private readonly changeEmitter = new vscode.EventEmitter<CyanNode | undefined>();
  private connected = false;
  private snapshot: JobSnapshot | undefined;
  private actionInFlight = false;

  public readonly onDidChangeTreeData = this.changeEmitter.event;

  // 更新连接和快照后刷新整个原生 Tree View
  public update(
    connected: boolean,
    snapshot: JobSnapshot | undefined,
    actionInFlight = false,
  ): void {
    this.connected = connected;
    this.snapshot = snapshot;
    this.actionInFlight = actionInFlight;
    this.changeEmitter.fire(undefined);
  }

  // 将内部节点映射为原生 TreeItem
  public getTreeItem(node: CyanNode): vscode.TreeItem {
    const collapsible = node.children === undefined
      ? vscode.TreeItemCollapsibleState.None
      : vscode.TreeItemCollapsibleState.Expanded;
    const item = new vscode.TreeItem(node.label, collapsible);
    item.description = node.description;
    item.tooltip = node.tooltip;
    item.contextValue = node.contextValue;
    item.command = node.command;
    if (node.icon !== undefined) {
      item.iconPath = new vscode.ThemeIcon(node.icon);
    }
    return item;
  }

  // 构造当前连接状态对应的最小层级
  public getChildren(node?: CyanNode): CyanNode[] {
    if (node !== undefined) {
      return node.children ?? [];
    }
    if (!this.connected) {
      return [
        {
          kind: "action",
          label: "Reconnect cyan core",
          command: { command: "cyan.reconnect", title: "Reconnect" },
          icon: "debug-disconnect",
        },
      ];
    }
    if (this.snapshot === undefined) {
      return [
        {
          kind: "action",
          label: "Start monitored training",
          command: { command: "cyan.startTraining", title: "Start" },
          icon: "play",
        },
      ];
    }
    const roots = [this.jobNode(this.snapshot)];
    if (this.snapshot.incident !== null && this.snapshot.incident !== undefined) {
      roots.push(this.incidentNode(this.snapshot));
    }
    return roots;
  }

  // 构造 Active Job 分组和上下文动作锚点
  private jobNode(snapshot: JobSnapshot): CyanNode {
    const executable = snapshot.argv[0] === undefined
      ? "training"
      : path.basename(snapshot.argv[0]);
    const running = isActiveJob(snapshot);
    return {
      kind: "job",
      label: "Active Job",
      description: `${executable} · ${snapshot.job.status}`,
      tooltip: snapshot.argv.join(" "),
      contextValue: running && !this.actionInFlight ? "cyan.job.running" : "cyan.job",
      icon: running ? "sync~spin" : "terminal",
      children: [
        {
          kind: "detail",
          label: snapshot.job.id,
          description: snapshot.attempt?.id,
          tooltip: snapshot.workspace_root,
        },
        {
          kind: "action",
          label: "Show Training Log",
          command: { command: "cyan.showLog", title: "Show Log" },
          icon: "terminal",
        },
      ],
    };
  }

  // 构造 Incident、diagnosis、evidence 与 proposal 层级
  private incidentNode(snapshot: JobSnapshot): CyanNode {
    const incident = snapshot.incident;
    if (incident === null || incident === undefined) {
      throw new Error("incident snapshot is missing");
    }
    const children: CyanNode[] = [];
    if (snapshot.diagnosis !== null && snapshot.diagnosis !== undefined) {
      children.push({
        kind: "diagnosis",
        label: snapshot.diagnosis.summary,
        description: snapshot.diagnosis.causal_support,
        tooltip: diagnosisDisposition(snapshot),
        command: { command: "cyan.openDiagnosis", title: "Open Diagnosis" },
        icon: "lightbulb",
        children: snapshot.diagnosis.evidence.map((evidence) => ({
          kind: "evidence",
          label: evidence.description,
          description: evidence.source,
          tooltip: evidence.reference,
          icon: "references",
        })),
      });
    } else {
      const noEvidence = incident.last_outcome === "smoke_evidence_unavailable";
      children.push({
        kind: "detail",
        label: noEvidence
          ? "Smoke failed; no output for re-diagnosis"
          : incident.status === "diagnosing"
            ? "Agent is investigating…"
            : incident.status,
        tooltip: noEvidence
          ? "The patch was rolled back, but Smoke produced no usable output."
          : undefined,
        icon: incident.status === "diagnosing" ? "sync~spin" : "info",
      });
    }
    if (snapshot.proposal !== null && snapshot.proposal !== undefined) {
      const awaiting = incident.status === "awaiting_approval";
      const applicable = awaiting && snapshot.can_apply === true;
      const pathLabel = snapshot.proposal.files[0]?.path ?? "proposal.diff";
      children.push({
        kind: "proposal",
        label: "Proposed fix",
        description: pathLabel,
        contextValue: this.actionInFlight
          ? "cyan.proposal.busy"
          : (applicable ? "cyan.proposal.applicable" : "cyan.proposal.reviewOnly"),
        command: { command: "cyan.reviewProposal", title: "Review Proposed Fix" },
        icon: "diff",
      });
    }
    return {
      kind: "incident",
      label: "Incident",
      description: incident.status,
      tooltip: incident.id,
      icon: incident.status === "resolved" ? "pass" : "warning",
      children,
    };
  }
}
