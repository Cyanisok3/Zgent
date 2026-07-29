import * as vscode from "vscode";

export class MemoryDocumentProvider implements vscode.TextDocumentContentProvider {
  private readonly documents = new Map<string, string>();
  private readonly changeEmitter = new vscode.EventEmitter<vscode.Uri>();

  public readonly onDidChange = this.changeEmitter.event;

  // 保存一份仅驻留扩展内存的只读文档并返回 URI
  public put(scheme: string, path: string, content: string): vscode.Uri {
    const uri = vscode.Uri.from({ scheme, path: `/${path.replace(/^\/+/, "")}` });
    this.documents.set(uri.toString(), content);
    this.changeEmitter.fire(uri);
    return uri;
  }

  // 向 VS Code 返回指定虚拟文档正文
  public provideTextDocumentContent(uri: vscode.Uri): string {
    return this.documents.get(uri.toString()) ?? "";
  }

  // 清除所有源码和诊断临时副本
  public clear(): void {
    this.documents.clear();
  }
}
