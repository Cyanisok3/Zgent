const READ_BYTES = 32 * 1024;

// 计算重新附着时的有界日志尾部起点
export function initialTailOffset(totalBytes: number, limit = READ_BYTES): number {
  return Math.max(0, totalBytes - limit);
}

// 将 Unix 换行规范化为 VS Code terminal 需要的 CRLF
export function terminalText(value: string): string {
  return value.replace(/\r?\n/g, "\r\n");
}
