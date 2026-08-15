import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";

interface ExtensionManifest {
  capabilities: {
    untrustedWorkspaces: { supported: boolean };
    virtualWorkspaces: boolean;
  };
  contributes: {
    menus: {
      commandPalette: Array<{ command: string; when: string }>;
    };
    views: unknown;
    configuration: unknown;
    [key: string]: unknown;
  };
}

// 读取扩展清单供静态产品边界测试复用
function manifest(): ExtensionManifest {
  const value = fs.readFileSync(path.resolve(__dirname, "../../package.json"), "utf8");
  return JSON.parse(value) as ExtensionManifest;
}

// 功能：验证 Alpha 只在可信的本地文件工作区启用
// 设计：直接检查 VS Code capabilities，避免以运行时提示代替平台门禁
test("manifest requires workspace trust and rejects virtual workspaces", () => {
  const value = manifest();
  assert.equal(value.capabilities.untrustedWorkspaces.supported, false);
  assert.equal(value.capabilities.virtualWorkspaces, false);
});

// 功能：验证审批、拒绝和取消只作为上下文动作而非全局命令
// 设计：检查 Command Palette 的显式隐藏项，锁定产品交互边界
test("state mutations are hidden from the command palette", () => {
  const hidden = new Map(
    manifest().contributes.menus.commandPalette.map((item) => [item.command, item.when]),
  );
  assert.equal(hidden.get("cyan.approveProposal"), "false");
  assert.equal(hidden.get("cyan.rejectProposal"), "false");
  assert.equal(hidden.get("cyan.cancelJob"), "false");
  assert.equal(hidden.get("cyan.recheckWorkflow"), "false");
});

// 功能：验证首版没有引入自定义 Webview surface
// 设计：静态拒绝 webview 类型贡献，保持原生 View 与虚拟文档方案
test("manifest contains no custom webview contribution", () => {
  const contributes = manifest().contributes;
  assert.equal("webviews" in contributes, false);
  assert.equal("customEditors" in contributes, false);
});
