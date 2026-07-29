import * as vscode from "vscode";
import { CyanController } from "./controller";
import { MemoryDocumentProvider } from "./documents";
import { CyanTreeProvider } from "./view";

let controller: CyanController | undefined;

// 激活薄型客户端并注册原生 VS Code surface
export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const tree = new CyanTreeProvider();
  const documents = new MemoryDocumentProvider();
  const statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    20,
  );
  statusBar.command = "cyan.showLog";
  const output = vscode.window.createOutputChannel("cyan");

  context.subscriptions.push(
    vscode.window.registerTreeDataProvider("cyan.training", tree),
    vscode.workspace.registerTextDocumentContentProvider("cyan-diagnosis", documents),
    vscode.workspace.registerTextDocumentContentProvider("cyan-before", documents),
    vscode.workspace.registerTextDocumentContentProvider("cyan-after", documents),
    statusBar,
    output,
  );

  try {
    controller = new CyanController(
      context,
      tree,
      documents,
      statusBar,
      output,
    );
  } catch (error) {
    output.appendLine(errorMessage(error));
    void vscode.window.showErrorMessage(errorMessage(error));
    return;
  }
  context.subscriptions.push(controller);

  // 将公开命令直接映射到 Controller 的上下文动作
  context.subscriptions.push(
    vscode.commands.registerCommand("cyan.startTraining", () => controller?.startTraining()),
    vscode.commands.registerCommand("cyan.attachJob", () => controller?.attachJob()),
    vscode.commands.registerCommand("cyan.showLog", () => controller?.showLog()),
    vscode.commands.registerCommand("cyan.openDiagnosis", () => controller?.openDiagnosis()),
    vscode.commands.registerCommand("cyan.reviewProposal", () => controller?.reviewProposal()),
    vscode.commands.registerCommand("cyan.approveProposal", () => controller?.approveProposal()),
    vscode.commands.registerCommand("cyan.rejectProposal", () => controller?.rejectProposal()),
    vscode.commands.registerCommand("cyan.cancelJob", () => controller?.cancelJob()),
    vscode.commands.registerCommand("cyan.refresh", () => controller?.refresh()),
    vscode.commands.registerCommand("cyan.reconnect", () => controller?.reconnect()),
  );

  await controller.initialize();
}

// 停用扩展时仅断开客户端，不停止 daemon
export function deactivate(): void {
  controller?.dispose();
  controller = undefined;
}

// 将未知异常收敛为短错误消息
function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
