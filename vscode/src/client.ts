import { spawn } from "node:child_process";
import * as net from "node:net";
import { EventEmitter } from "node:events";
import { CoreStartResult, JsonObject } from "./types";

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
}

export class RpcError extends Error {
  public readonly code: number;

  // 保存 daemon 返回的结构化错误码
  public constructor(code: number, message: string) {
    super(message);
    this.code = code;
  }
}

export class RpcClient {
  private socket: net.Socket | undefined;
  private buffer = "";
  private nextId = 1;
  private readonly pending = new Map<string, PendingRequest>();
  private readonly emitter = new EventEmitter();
  private closed = true;

  // 建立一次 loopback NDJSON 连接
  public async connect(host: string, port: number): Promise<void> {
    if (this.socket !== undefined) {
      return;
    }
    await new Promise<void>((resolve, reject) => {
      const socket = net.createConnection({ host, port });
      const fail = (error: Error): void => {
        socket.destroy();
        reject(error);
      };
      socket.once("error", fail);
      socket.once("connect", () => {
        socket.off("error", fail);
        this.socket = socket;
        this.closed = false;
        socket.setEncoding("utf8");
        socket.on("data", (chunk: string) => this.consume(chunk));
        socket.on("error", (error) => this.finish(error));
        socket.on("close", () => this.finish(new Error("cyan core disconnected")));
        resolve();
      });
    });
  }

  // 发送一个 JSON-RPC 请求并等待对应响应
  public async request<T>(method: string, params: JsonObject): Promise<T> {
    const socket = this.socket;
    if (socket === undefined || this.closed) {
      throw new Error("cyan core is not connected");
    }
    const id = `vscode-${this.nextId++}`;
    const response = new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (value) => resolve(value as T),
        reject,
      });
    });
    socket.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
    return response;
  }

  // 订阅 daemon 主动推送的事件
  public onEvent(listener: (event: JsonObject) => void): () => void {
    this.emitter.on("event", listener);
    return () => this.emitter.off("event", listener);
  }

  // 订阅连接关闭通知
  public onClose(listener: (error: Error) => void): () => void {
    this.emitter.on("close", listener);
    return () => this.emitter.off("close", listener);
  }

  // 主动关闭客户端连接但不停止 daemon
  public dispose(): void {
    this.finish(new Error("cyan client disposed"));
    this.socket?.destroy();
    this.socket = undefined;
  }

  // 消费完整 NDJSON 帧并分派响应或事件
  private consume(chunk: string): void {
    this.buffer += chunk;
    while (true) {
      const newline = this.buffer.indexOf("\n");
      if (newline < 0) {
        return;
      }
      const line = this.buffer.slice(0, newline);
      this.buffer = this.buffer.slice(newline + 1);
      if (line.length === 0) {
        continue;
      }
      let message: JsonObject;
      try {
        message = JSON.parse(line) as JsonObject;
      } catch {
        this.finish(new Error("cyan core sent invalid JSON"));
        return;
      }
      if (message.kind === "event" && typeof message.event === "object" && message.event !== null) {
        this.emitter.emit("event", message.event as JsonObject);
        continue;
      }
      const id = typeof message.id === "string" ? message.id : undefined;
      if (id === undefined) {
        continue;
      }
      const pending = this.pending.get(id);
      if (pending === undefined) {
        continue;
      }
      this.pending.delete(id);
      const error = message.error as JsonObject | undefined;
      if (error !== undefined) {
        pending.reject(new RpcError(Number(error.code), String(error.message)));
      } else {
        pending.resolve(message.result);
      }
    }
  }

  // 结束连接并拒绝所有未完成请求
  private finish(error: Error): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    for (const request of this.pending.values()) {
      request.reject(error);
    }
    this.pending.clear();
    this.emitter.emit("close", error);
  }
}

// 不经 shell 启动或发现 cyan daemon 并解析单行 JSON
export async function startCore(
  executable: string,
  workspaceRoot: string,
): Promise<CoreStartResult> {
  return new Promise<CoreStartResult>((resolve, reject) => {
    const process = spawn(executable, ["core", "start", "--json"], {
      cwd: workspaceRoot,
      env: globalThis.process.env,
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    process.stdout.setEncoding("utf8");
    process.stderr.setEncoding("utf8");
    process.stdout.on("data", (chunk: string) => {
      if (stdout.length < 64 * 1024) {
        stdout += chunk;
      }
    });
    process.stderr.on("data", (chunk: string) => {
      if (stderr.length < 64 * 1024) {
        stderr += chunk;
      }
    });
    process.once("error", reject);
    process.once("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr.trim() || `cyan core start exited with status ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout.trim()) as CoreStartResult);
      } catch {
        reject(new Error("cyan core start did not return valid JSON"));
      }
    });
  });
}

// 复制扩展宿主的字符串环境供 harness 构造精确 LaunchSpec
export function processEnvironment(): Record<string, string> {
  const environment: Record<string, string> = {};
  for (const [name, value] of Object.entries(globalThis.process.env)) {
    if (value !== undefined) {
      environment[name] = value;
    }
  }
  return environment;
}
