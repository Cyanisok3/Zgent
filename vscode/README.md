# cyan for VS Code

Thin local VS Code client for the cyan ML training Incident Agent.

## Alpha boundary

- macOS or Linux;
- one trusted local workspace folder;
- an installed `cyan` executable;
- no Marketplace distribution, remote workspace, multi-root workspace, Windows, or chat UI.

Set `cyan.executablePath` when `cyan` is not available on the VS Code extension host `PATH`.

## Flow

1. Open the ML project root and trust the workspace.
2. Open the cyan Activity Bar view.
3. Select **Start monitored training**, paste one logical training command, and review the deterministic preview.
4. Read the real training output in the bottom `cyan` terminal.
5. When an Incident is ready, open diagnosis evidence and recovery. Review and approve a patch, or complete the listed operator actions and select **Recheck Workflow**.

The preview and Job tree show the optional frozen Workflow Contract summary and current
preflight/main/postflight phase. VS Code never parses or persists workflow state itself.

Closing VS Code or the cyan terminal only detaches the client. It does not cancel the daemon-owned training process.

## Development

```bash
npm install
npm run lint
npm run test:unit
npm run test:extension
npm run package
code --install-extension cyan-vscode-0.0.1.vsix
```
