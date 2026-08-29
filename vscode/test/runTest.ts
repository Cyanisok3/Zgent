import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";
import { runTests } from "@vscode/test-electron";

// 选择本机 VS Code 二进制，找不到时交给测试库下载稳定版本
function resolveVsCodeExecutable(): string | undefined {
  const configured = process.env.VSCODE_EXECUTABLE;
  if (configured !== undefined && fs.existsSync(configured)) {
    return configured;
  }
  if (process.platform !== "darwin") {
    return undefined;
  }
  const candidates = [
    "/Applications/Visual Studio Code.app/Contents/MacOS/Electron",
    "/Applications/Visual Studio Code.app/Contents/MacOS/Code",
  ];
  return candidates.find((candidate) => fs.existsSync(candidate));
}

// 向操作系统申请一个临时 loopback 端口
async function freePort(): Promise<number> {
  return new Promise<number>((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (address === null || typeof address === "string") {
        server.close();
        reject(new Error("failed to allocate test port"));
        return;
      }
      const port = address.port;
      server.close(() => resolve(port));
    });
  });
}

// 启动真实 VS Code Extension Host 并在临时工作区运行 smoke suite
async function main(): Promise<void> {
  const extensionDevelopmentPath = path.resolve(__dirname, "../..");
  const projectRoot = path.resolve(extensionDevelopmentPath, "..");
  const cyanExecutable = path.join(projectRoot, ".venv", "bin", "cyan");
  const vscodeExecutablePath = resolveVsCodeExecutable();
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "cyan-vscode-test-"));
  const workspace = path.join(temporary, "workspace");
  const testHome = path.join(temporary, "home");
  fs.mkdirSync(path.join(workspace, ".cyan"), { recursive: true });
  fs.mkdirSync(testHome, { recursive: true });
  const port = await freePort();
  fs.writeFileSync(
    path.join(workspace, ".cyan", "config.toml"),
    `[core]\nhost = "127.0.0.1"\nport = ${port}\n\n[trace]\nenabled = false\n`,
    "utf8",
  );
  const environment: NodeJS.ProcessEnv = {
    ...process.env,
    HOME: testHome,
    CYAN_EXECUTABLE: cyanExecutable,
  };
  delete environment.ELECTRON_RUN_AS_NODE;
  delete environment.VSCODE_ESM_ENTRYPOINT;
  delete process.env.ELECTRON_RUN_AS_NODE;
  delete process.env.VSCODE_ESM_ENTRYPOINT;
  try {
    const testOptions: Parameters<typeof runTests>[0] = {
      extensionDevelopmentPath,
      extensionTestsPath: path.resolve(__dirname, "suite", "index"),
      extensionTestsEnv: environment,
      launchArgs: [
        workspace,
        "--disable-extensions",
        "--disable-workspace-trust",
        "--skip-welcome",
        `--user-data-dir=${path.join(temporary, "user-data")}`,
        `--extensions-dir=${path.join(temporary, "extensions")}`,
      ],
    };
    if (vscodeExecutablePath !== undefined) {
      testOptions.vscodeExecutablePath = vscodeExecutablePath;
    }
    await runTests(testOptions);
  } finally {
    spawnSync(cyanExecutable, ["core", "stop"], {
      cwd: workspace,
      env: environment,
      stdio: "ignore",
    });
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}

void main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
