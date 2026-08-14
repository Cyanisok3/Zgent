import * as vscode from "vscode";
import { RpcClient } from "./client";
import { initialTailOffset, terminalText } from "./terminalSupport";
import { LogChunk } from "./types";

const READ_BYTES = 32 * 1024;
const POLL_MS = 300;

export class TrainingPseudoterminal implements vscode.Pseudoterminal {
  private readonly writeEmitter = new vscode.EventEmitter<string>();
  private readonly closeEmitter = new vscode.EventEmitter<void>();
  private attemptId: string;
  private cursors: Record<"stdout" | "stderr", number> = { stdout: 0, stderr: 0 };
  private initialized = false;
  private disposed = false;
  private timer: NodeJS.Timeout | undefined;
  private inputNoticeShown = false;

  public readonly onDidWrite = this.writeEmitter.event;
  public readonly onDidClose = this.closeEmitter.event;

  // 绑定一个 Job Attempt 的只读日志流
  public constructor(
    private readonly client: RpcClient,
    private readonly jobId: string,
    attemptId: string,
  ) {
    this.attemptId = attemptId;
  }

  // 打开伪终端并从日志尾部开始持续读取
  public open(): void {
    this.writeEmitter.fire("\x1b[2mcyan read-only training log · close detaches only\x1b[0m\r\n");
    void this.poll();
  }

  // 关闭本地日志镜像但不取消 daemon Job
  public close(): void {
    this.disposed = true;
    if (this.timer !== undefined) {
      clearTimeout(this.timer);
    }
  }

  // 提醒用户训练进程不接受终端输入
  public handleInput(): void {
    if (!this.inputNoticeShown) {
      this.inputNoticeShown = true;
      this.writeEmitter.fire("\r\n\x1b[2mThis monitor is read-only; use the Cancel action in cyan.\x1b[0m\r\n");
    }
  }

  // 在原命令重跑后切换到新的 Attempt 日志
  public setAttempt(attemptId: string): void {
    if (attemptId === this.attemptId) {
      return;
    }
    this.attemptId = attemptId;
    this.cursors = { stdout: 0, stderr: 0 };
    this.initialized = false;
    this.writeEmitter.fire("\r\n\x1b[2m—— cyan retry attempt ——\x1b[0m\r\n");
    void this.poll();
  }

  // 读取两个原始日志流并推进各自字节游标
  private async readAvailable(): Promise<void> {
    const attemptId = this.attemptId;
    if (!this.initialized) {
      for (const stream of ["stdout", "stderr"] as const) {
        const head = await this.client.request<LogChunk>("job.read_log", {
          job_id: this.jobId,
          attempt_id: attemptId,
          stream,
          offset: 0,
          limit: 1,
        });
        this.cursors[stream] = initialTailOffset(head.total_bytes);
      }
      this.initialized = true;
    }
    for (const stream of ["stdout", "stderr"] as const) {
      const chunk = await this.client.request<LogChunk>("job.read_log", {
        job_id: this.jobId,
        attempt_id: attemptId,
        stream,
        offset: this.cursors[stream],
        limit: READ_BYTES,
      });
      if (attemptId !== this.attemptId) {
        return;
      }
      this.cursors[stream] = chunk.next_offset;
      if (chunk.data.length > 0) {
        this.writeEmitter.fire(terminalText(chunk.data));
      }
    }
  }

  // 串行轮询日志，避免慢 RPC 产生重叠读取
  private async poll(): Promise<void> {
    if (this.disposed || this.timer !== undefined) {
      return;
    }
    try {
      await this.readAvailable();
    } catch (error) {
      if (!this.disposed) {
        this.writeEmitter.fire(`\r\n\x1b[31mcyan log error: ${errorMessage(error)}\x1b[0m\r\n`);
      }
    }
    if (!this.disposed) {
      this.timer = setTimeout(() => {
        this.timer = undefined;
        void this.poll();
      }, POLL_MS);
    }
  }
}

// 将未知异常收敛为不含调用参数的短消息
function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
