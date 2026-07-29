import * as fs from "node:fs";
import * as path from "node:path";
import * as vscode from "vscode";
import { MemoryDocumentProvider } from "./documents";
import { processEnvironment, RpcClient, startCore } from "./client";
import { TrainingPseudoterminal } from "./terminal";
import {
  IncidentReview,
  JobSnapshot,
  LaunchPreview,
  PongResult,
  WIRE_PROTOCOL_VERSION,
  diagnosisMarkdown,
  isActiveJob,
  statusPresentation,
} from "./types";
import { CyanTreeProvider } from "./view";

const ATTACHED_JOB_KEY = "cyan.attachedJobId";

interface CoreConnection {
  client: RpcClient;
  host: string;
  port: number;
}

interface TerminalHandle {
  jobId: string;
  pty: TrainingPseudoterminal;
  terminal: vscode.Terminal;
  closeListener: vscode.Disposable;
}

interface PreviewChoice extends vscode.QuickPickItem {
  action: "start" | "edit";
}

export class CyanController implements vscode.Disposable {
  private readonly workspaceRoot: string;
  private connection: CoreConnection | undefined;
  private snapshot: JobSnapshot | undefined;
  private terminalHandle: TerminalHandle | undefined;
  private disposed = false;
  private reconnectTimer: NodeJS.Timeout | undefined;
  private refreshPending = false;
  private actionInFlight = false;
  private firstSnapshot = true;
  private previousIncidentStatus: string | undefined;

  // 绑定单工作区所需的原生 VS Code 服务
  public constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly tree: CyanTreeProvider,
    private readonly documents: MemoryDocumentProvider,
    private readonly statusBar: vscode.StatusBarItem,
    private readonly output: vscode.OutputChannel,
  ) {
    const folder = vscode.workspace.workspaceFolders?.[0];
    if (folder === undefined) {
      throw new Error("Open one local ML project folder before using cyan.");
    }
    this.workspaceRoot = fs.realpathSync.native(folder.uri.fsPath);
  }

  // 连接 daemon 并恢复当前工作区的唯一任务
  public async initialize(): Promise<void> {
    if (process.platform !== "darwin" && process.platform !== "linux") {
      throw new Error("cyan VSIX Alpha currently supports macOS and Linux.");
    }
    const folders = vscode.workspace.workspaceFolders ?? [];
    if (folders.length !== 1 || folders[0]?.uri.scheme !== "file") {
      throw new Error("cyan VSIX Alpha requires one local workspace folder.");
    }
    if (!vscode.workspace.isTrusted) {
      throw new Error("Trust this workspace before starting training code.");
    }
    await this.connect();
  }

  // 释放插件资源但保持 daemon 和训练进程运行
  public dispose(): void {
    this.disposed = true;
    if (this.reconnectTimer !== undefined) {
      clearTimeout(this.reconnectTimer);
    }
    this.connection?.client.dispose();
    this.connection = undefined;
    this.terminalHandle?.pty.close();
    this.terminalHandle?.closeListener.dispose();
    this.terminalHandle = undefined;
    this.documents.clear();
  }

  // 打开命令输入、确定性预览和显式启动三步流程
  public async startTraining(): Promise<void> {
    if (this.connection === undefined) {
      await this.connect();
    }
    if (isActiveJob(this.snapshot)) {
      void vscode.window.showWarningMessage(
        "Finish or cancel the attached training process before starting another.",
      );
      return;
    }
    let initialValue = "";
    while (true) {
      const command = await vscode.window.showInputBox({
        title: "Start monitored training",
        prompt: "Paste one logical training command. Wrap complex shell logic in a script.",
        placeHolder: "python train.py --config configs/local.yaml",
        value: initialValue,
        ignoreFocusOut: true,
        validateInput: (value) => (
          value.length > 64 * 1024 ? "Training command exceeds 64 KiB." : undefined
        ),
      });
      if (command === undefined) {
        return;
      }
      initialValue = command;
      let preview: LaunchPreview;
      try {
        preview = await this.request<LaunchPreview>("launch.preview", {
          command,
          workspace_root: this.workspaceRoot,
          env: processEnvironment(),
        });
      } catch (error) {
        void vscode.window.showErrorMessage(`Invalid training command: ${errorMessage(error)}`);
        continue;
      }
      const choice = await vscode.window.showQuickPick<PreviewChoice>(
        [
          {
            label: "$(play) Start training",
            action: "start",
            description: preview.cwd,
            detail: previewDetail(preview),
          },
          {
            label: "$(edit) Edit command",
            action: "edit",
            detail: "Return to the command input without starting a process.",
          },
        ],
        {
          title: "Confirm cyan training launch",
          placeHolder: "Review the deterministic LaunchSpec preview",
          ignoreFocusOut: true,
        },
      );
      if (choice === undefined) {
        return;
      }
      if (choice.action === "edit") {
        continue;
      }
      try {
        const result = await this.request<{ job_id: string }>("launch.start", {
          command,
          workspace_root: this.workspaceRoot,
          env: processEnvironment(),
          preview_fingerprint: preview.fingerprint,
        });
        await this.attach(result.job_id, false);
        await this.showLog();
      } catch (error) {
        void vscode.window.showErrorMessage(`Training did not start: ${errorMessage(error)}`);
      }
      return;
    }
  }

  // 让用户从当前工作区历史任务中显式选择附着目标
  public async attachJob(): Promise<void> {
    const jobs = await this.workspaceJobs();
    if (jobs.length === 0) {
      void vscode.window.showInformationMessage("No cyan jobs exist in this workspace.");
      return;
    }
    const selected = await vscode.window.showQuickPick(
      jobs.map((job) => ({
        label: `${job.job.status} · ${job.argv[0] ?? "training"}`,
        description: job.job.id,
        detail: job.argv.join(" "),
        job,
      })),
      {
        title: "Attach to a cyan Job",
        placeHolder: "Select one persisted Job",
      },
    );
    if (selected !== undefined) {
      await this.attach(selected.job.job.id, true);
    }
  }

  // 展示或创建当前 Attempt 的只读 Pseudoterminal
  public async showLog(): Promise<void> {
    const snapshot = this.snapshot;
    const attemptId = snapshot?.attempt?.id;
    if (snapshot === undefined || attemptId === undefined) {
      void vscode.window.showWarningMessage("The attached Job has no training attempt.");
      return;
    }
    if (
      this.terminalHandle !== undefined
      && this.terminalHandle.jobId === snapshot.job.id
    ) {
      this.terminalHandle.pty.setAttempt(attemptId);
      this.terminalHandle.terminal.show();
      return;
    }
    this.closeTerminalMirror();
    const connection = this.requireConnection();
    const pty = new TrainingPseudoterminal(connection.client, snapshot.job.id, attemptId);
    const terminal = vscode.window.createTerminal({
      name: `cyan · ${path.basename(snapshot.argv[0] ?? "training")}`,
      pty,
      isTransient: true,
    });
    const closeListener = vscode.window.onDidCloseTerminal((closed) => {
      if (closed === terminal) {
        pty.close();
        closeListener.dispose();
        if (this.terminalHandle?.terminal === terminal) {
          this.terminalHandle = undefined;
        }
      }
    });
    this.terminalHandle = {
      jobId: snapshot.job.id,
      pty,
      terminal,
      closeListener,
    };
    terminal.show();
  }

  // 在内置 Markdown Preview 中展示完整 diagnosis 和 evidence
  public async openDiagnosis(): Promise<void> {
    if (this.snapshot === undefined) {
      return;
    }
    const uri = this.documents.put(
      "cyan-diagnosis",
      `${this.snapshot.job.id}/diagnosis.md`,
      diagnosisMarkdown(this.snapshot),
    );
    await vscode.commands.executeCommand("markdown.showPreview", uri);
  }

  // 从 backend 取得前后文本并调用原生 vscode.diff
  public async reviewProposal(): Promise<void> {
    const snapshot = this.requireProposal();
    const incident = snapshot.incident;
    const proposal = snapshot.proposal;
    if (incident === null || incident === undefined || proposal === null || proposal === undefined) {
      return;
    }
    try {
      const review = await this.request<IncidentReview>("incident.review", {
        job_id: snapshot.job.id,
        incident_id: incident.id,
        proposal_id: proposal.id,
      });
      const safePath = review.path.replace(/^\/+/, "");
      const before = this.documents.put(
        "cyan-before",
        `${review.proposal_id}/${safePath}`,
        review.before_text,
      );
      const after = this.documents.put(
        "cyan-after",
        `${review.proposal_id}/${safePath}`,
        review.after_text,
      );
      await vscode.commands.executeCommand(
        "vscode.diff",
        before,
        after,
        `${review.path} — current ↔ cyan proposal`,
        { preview: true },
      );
    } catch (error) {
      void vscode.window.showErrorMessage(`Proposal review failed: ${errorMessage(error)}`);
    }
  }

  // 根据 smoke 配置提交一次明确批准
  public async approveProposal(): Promise<void> {
    const snapshot = this.requireProposal();
    if (snapshot.can_apply !== true || this.actionInFlight) {
      return;
    }
    let runSmoke = false;
    if (snapshot.smoke_config !== null && snapshot.smoke_config !== undefined) {
      const choice = await vscode.window.showQuickPick(
        [
          {
            label: "Approve and run smoke",
            runSmoke: true,
            description: "Recommended",
          },
          {
            label: "Approve without smoke",
            runSmoke: false,
          },
        ],
        {
          title: "Approve cyan proposed fix",
          placeHolder: "The original training command remains the final verifier",
        },
      );
      if (choice === undefined) {
        return;
      }
      runSmoke = choice.runSmoke;
    }
    await this.decide("approve", runSmoke);
  }

  // 提交一次明确拒绝并保持工作区未修改
  public async rejectProposal(): Promise<void> {
    if (this.actionInFlight) {
      return;
    }
    await this.decide("reject", false);
  }

  // 直接请求 JobSupervisor 取消当前运行进程组
  public async cancelJob(): Promise<void> {
    if (!isActiveJob(this.snapshot) || this.actionInFlight) {
      return;
    }
    this.actionInFlight = true;
    this.updatePresentation();
    try {
      await this.request("job.cancel", { job_id: this.snapshot?.job.id });
    } catch (error) {
      void vscode.window.showErrorMessage(`Training cancellation failed: ${errorMessage(error)}`);
    } finally {
      this.actionInFlight = false;
      await this.refresh();
    }
  }

  // 从 daemon 真相源重新读取当前 Job snapshot
  public async refresh(): Promise<void> {
    const jobId = this.snapshot?.job.id
      ?? this.context.workspaceState.get<string>(ATTACHED_JOB_KEY);
    if (jobId === undefined || this.connection === undefined) {
      this.updatePresentation();
      return;
    }
    try {
      const snapshot = await this.request<JobSnapshot>("job.get", { job_id: jobId });
      if (snapshot.workspace_root !== this.workspaceRoot) {
        throw new Error("Attached Job belongs to another workspace.");
      }
      const previousAttempt = this.snapshot?.attempt?.id;
      this.snapshot = snapshot;
      if (
        this.terminalHandle !== undefined
        && snapshot.attempt?.id !== undefined
        && snapshot.attempt.id !== previousAttempt
      ) {
        this.terminalHandle.pty.setAttempt(snapshot.attempt.id);
      }
      this.notifyTransition(snapshot);
      this.updatePresentation();
    } catch (error) {
      this.output.appendLine(`snapshot refresh failed: ${errorMessage(error)}`);
    }
  }

  // 主动丢弃旧 socket 并重新执行 daemon 发现
  public async reconnect(): Promise<void> {
    if (this.reconnectTimer !== undefined) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = undefined;
    }
    this.connection?.client.dispose();
    this.connection = undefined;
    await this.connect();
  }

  // 启动 cyan、校验协议和工作区并订阅状态事件
  private async connect(): Promise<void> {
    if (this.disposed || this.connection !== undefined) {
      return;
    }
    const executable = vscode.workspace
      .getConfiguration("cyan")
      .get<string>("executablePath", "")
      .trim() || "cyan";
    try {
      const endpoint = await startCore(executable, this.workspaceRoot);
      const client = new RpcClient();
      await client.connect(endpoint.host, endpoint.port);
      const pong = await client.request<PongResult>("core.ping", {
        client: "cyan-vscode/0.0.1",
      });
      if (pong.protocol_version !== WIRE_PROTOCOL_VERSION) {
        client.dispose();
        throw new Error(
          `Wire protocol ${pong.protocol_version} is incompatible with ${WIRE_PROTOCOL_VERSION}.`,
        );
      }
      if (pong.startup_workspace_root !== this.workspaceRoot) {
        const jobs = await client.request<{ jobs: JobSnapshot[] }>("job.list", {});
        const active = jobs.jobs.filter(isActiveJob);
        if (active.length > 0) {
          client.dispose();
          throw new Error(
            `cyan core is supervising ${pong.startup_workspace_root}; finish that Job before switching.`,
          );
        }
        const action = await vscode.window.showWarningMessage(
          `cyan core was started for ${pong.startup_workspace_root}.`,
          "Restart Core for This Workspace",
        );
        if (action !== "Restart Core for This Workspace") {
          client.dispose();
          throw new Error("cyan core workspace does not match the open folder.");
        }
        await client.request("core.shutdown", {});
        client.dispose();
        return await this.connect();
      }
      this.connection = { client, host: endpoint.host, port: endpoint.port };
      client.onEvent(() => this.scheduleRefresh());
      client.onClose((error) => this.handleDisconnect(error));
      await client.request("event.subscribe", {
        topics: ["job.*", "incident.*"],
        scope: "global",
        after_seq: 0,
      });
      await this.restoreJob();
      this.updatePresentation();
    } catch (error) {
      this.connection = undefined;
      this.updatePresentation();
      this.output.appendLine(`connection failed: ${errorMessage(error)}`);
      void vscode.window.showErrorMessage(`cyan connection failed: ${errorMessage(error)}`);
    }
  }

  // 从持久化 job_id 或唯一工作区 Job 恢复附着
  private async restoreJob(): Promise<void> {
    const jobs = await this.workspaceJobs();
    const stored = this.context.workspaceState.get<string>(ATTACHED_JOB_KEY);
    const storedJob = jobs.find((job) => job.job.id === stored);
    if (storedJob !== undefined) {
      this.snapshot = storedJob;
    } else {
      const active = jobs.filter(isActiveJob);
      this.snapshot = active.length === 1
        ? active[0]
        : (jobs.length === 1 ? jobs[0] : undefined);
    }
    if (this.snapshot !== undefined) {
      await this.context.workspaceState.update(ATTACHED_JOB_KEY, this.snapshot.job.id);
    }
    this.firstSnapshot = true;
    await this.refresh();
  }

  // 返回当前工作区内的安全 Job 视图
  private async workspaceJobs(): Promise<JobSnapshot[]> {
    const result = await this.request<{ jobs: JobSnapshot[] }>("job.list", {});
    return result.jobs.filter((job) => job.workspace_root === this.workspaceRoot);
  }

  // 保存 attached job_id 并从 daemon 刷新完整快照
  private async attach(jobId: string, tailLogs: boolean): Promise<void> {
    await this.context.workspaceState.update(ATTACHED_JOB_KEY, jobId);
    this.snapshot = undefined;
    this.firstSnapshot = true;
    await this.refresh();
    if (tailLogs) {
      await this.showLog();
    }
  }

  // 调用 incident.decide 并在失败后以真实 snapshot 恢复 UI
  private async decide(decision: "approve" | "reject", runSmoke: boolean): Promise<void> {
    const snapshot = this.requireProposal();
    const incident = snapshot.incident;
    const proposal = snapshot.proposal;
    if (incident === null || incident === undefined || proposal === null || proposal === undefined) {
      return;
    }
    this.actionInFlight = true;
    this.updatePresentation();
    try {
      await this.request("incident.decide", {
        incident_id: incident.id,
        proposal_id: proposal.id,
        decision,
        run_smoke: runSmoke,
        smoke_config_fingerprint: snapshot.smoke_config_fingerprint ?? null,
      });
    } catch (error) {
      void vscode.window.showErrorMessage(`Incident decision failed: ${errorMessage(error)}`);
    } finally {
      this.actionInFlight = false;
      await this.refresh();
    }
  }

  // 确认当前快照包含可操作的 proposal
  private requireProposal(): JobSnapshot {
    if (
      this.snapshot === undefined
      || this.snapshot.incident === null
      || this.snapshot.incident === undefined
      || this.snapshot.proposal === null
      || this.snapshot.proposal === undefined
    ) {
      throw new Error("No proposal is available.");
    }
    return this.snapshot;
  }

  // 返回活动 RPC 连接或抛出可读错误
  private requireConnection(): CoreConnection {
    if (this.connection === undefined) {
      throw new Error("cyan core is not connected");
    }
    return this.connection;
  }

  // 通过当前连接发送一个结构化命令
  private async request<T = unknown>(
    method: string,
    params: Record<string, unknown>,
  ): Promise<T> {
    return this.requireConnection().client.request<T>(method, params);
  }

  // 收敛短时间内的多个状态事件为一次 snapshot 刷新
  private scheduleRefresh(): void {
    if (this.refreshPending) {
      return;
    }
    this.refreshPending = true;
    setTimeout(() => {
      this.refreshPending = false;
      void this.refresh();
    }, 50);
  }

  // 断线后清理前端连接并安排一次有界重连
  private handleDisconnect(error: Error): void {
    if (this.disposed) {
      return;
    }
    this.output.appendLine(`disconnected: ${error.message}`);
    this.connection = undefined;
    this.updatePresentation();
    if (this.reconnectTimer === undefined) {
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = undefined;
        void this.connect();
      }, 1000);
    }
  }

  // 仅在关键 Incident 状态变化时发送用户通知
  private notifyTransition(snapshot: JobSnapshot): void {
    const current = snapshot.incident?.status;
    if (this.firstSnapshot) {
      this.firstSnapshot = false;
      this.previousIncidentStatus = current;
      return;
    }
    if (current === this.previousIncidentStatus) {
      return;
    }
    this.previousIncidentStatus = current;
    if (current === "diagnosing") {
      void vscode.window.showInformationMessage(
        "Training failed. cyan is investigating the Incident.",
      );
    } else if (current === "awaiting_approval") {
      void vscode.window.showInformationMessage(
        "cyan diagnosis and proposed fix are ready for review.",
        "Review",
      ).then((action) => {
        if (action === "Review") {
          void this.reviewProposal();
        }
      });
    } else if (current === "resolved") {
      void vscode.window.showInformationMessage(
        "cyan resolved the Incident; the original command succeeded.",
      );
    } else if (current === "unresolved" || current === "rollback_blocked") {
      void vscode.window.showWarningMessage(`cyan Incident remains ${current}.`);
    }
  }

  // 同步 Tree View、状态栏和菜单 context keys
  private updatePresentation(): void {
    const connected = this.connection !== undefined;
    this.tree.update(connected, this.snapshot, this.actionInFlight);
    const presentation = statusPresentation(connected, this.snapshot);
    this.statusBar.text = `$(${presentation.icon}) cyan: ${presentation.text}`;
    this.statusBar.tooltip = presentation.tooltip;
    this.statusBar.show();
    void vscode.commands.executeCommand("setContext", "cyan.connected", connected);
    void vscode.commands.executeCommand("setContext", "cyan.hasJob", this.snapshot !== undefined);
    void vscode.commands.executeCommand(
      "setContext",
      "cyan.canStart",
      connected && !isActiveJob(this.snapshot),
    );
  }

  // 关闭现有日志镜像并解除 VS Code 监听器
  private closeTerminalMirror(): void {
    this.terminalHandle?.pty.close();
    this.terminalHandle?.terminal.dispose();
    this.terminalHandle?.closeListener.dispose();
    this.terminalHandle = undefined;
  }
}

// 将 LaunchPreview 压缩为 Quick Pick 的单行详情
function previewDetail(preview: LaunchPreview): string {
  const overrides = Object.keys(preview.env_overrides).length === 0
    ? "env overrides: none"
    : `env overrides: ${Object.entries(preview.env_overrides)
      .map(([name, value]) => `${name}=${value}`)
      .join(", ")}`;
  const configs = preview.config_paths.length === 0
    ? "config paths: none"
    : `config paths: ${preview.config_paths.join(", ")}`;
  return `${preview.executable} · argv: ${preview.argv.join(" ")} · ${overrides} · ${configs}`;
}

// 将未知异常收敛为短错误消息
function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
