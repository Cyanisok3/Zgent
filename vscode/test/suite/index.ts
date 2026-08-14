import * as assert from "node:assert/strict";
import * as vscode from "vscode";

// 激活真实扩展并验证核心命令可在 Extension Host 中调用
export async function run(): Promise<void> {
  const extension = vscode.extensions.getExtension("cyan-local.cyan-vscode");
  assert.notEqual(extension, undefined);
  const executable = process.env.CYAN_EXECUTABLE;
  assert.notEqual(executable, undefined);
  await vscode.workspace
    .getConfiguration("cyan")
    .update("executablePath", executable, vscode.ConfigurationTarget.Global);
  await extension?.activate();
  assert.equal(extension?.isActive, true);

  const commands = await vscode.commands.getCommands(true);
  for (const command of [
    "cyan.startTraining",
    "cyan.attachJob",
    "cyan.showLog",
    "cyan.reviewProposal",
    "cyan.approveProposal",
    "cyan.rejectProposal",
    "cyan.cancelJob",
  ]) {
    assert.equal(commands.includes(command), true, `${command} is not registered`);
  }
  await vscode.commands.executeCommand("cyan.refresh");
}
