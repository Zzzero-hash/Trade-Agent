# MCP Tooling Overview

The `mcp_tools` workspace now hosts multiple Model Context Protocol servers
that we rely on during development. This document explains what is available
and how to run each server against our local infrastructure.

## Python: Trading Platform Helper Server

Path: `mcp_tools/trading_platform_server.py`

This is an internal stdio MCP server that surfaces repository-specific
shortcuts (roadmap tasks, spec lookups, TODO search). Launch it from the repo
root with:

```bash
python -m mcp_tools.trading_platform_server
```

## Kubernetes Server (flux159/mcp-server-kubernetes)

Path: `mcp_tools/mcp-server-kubernetes`

We vendor the upstream project and install its dependencies via `npm` so we
can run the server directly from source. Prerequisites:

- `kubectl` and `helm` on your `PATH`
- Access to the `docker-desktop` context (or any other kube context you want
  to operate on)

Initial setup (already executed, repeat only if you blow away `node_modules`):

```bash
cd mcp_tools/mcp-server-kubernetes
npm install
npm run build
```

Start the server over stdio (ideal when wiring it into MCP-capable tools):

```bash
cd mcp_tools/mcp-server-kubernetes
node dist/index.js
```

You can also rely on the published package without leaving our repo:

```bash
npx --yes mcp-server-kubernetes
```

Helpful smoke tests while the server is running:

- `kubectl get pods -n ai-trading-platform` verifies the kubectl context you
  expose to the server is healthy.
- The server logs a handshake message ("Starting Kubernetes MCP server …")
  when it is ready to accept MCP connections.

For advanced authentication options (multiple kubeconfigs, token-based auth,
etc.) refer to `mcp_tools/mcp-server-kubernetes/ADVANCED_README.md`.

## Context7 Server (@upstash/context7-mcp)

Path: `mcp_tools/context7`

Context7 supplies up-to-date documentation snippets for popular libraries.
We install dependencies and compile the TypeScript sources locally so the
server can be executed straight from source.

Setup (already executed):

```bash
cd mcp_tools/context7
npm install
npx --yes tsc
npx --yes shx chmod +x dist/index.js  # no-op on Windows, safe elsewhere
```

Run the server:

```bash
cd mcp_tools/context7
node dist/index.js --transport stdio
```

When you only need the published build, skip the local checkout and invoke:

```bash
npx --yes @upstash/context7-mcp
```

### API keys

Anonymous usage works for light requests. For higher rate limits export your
Context7 API token before starting the server:

```bash
export CONTEXT7_API_KEY="<token>"      # bash
$Env:CONTEXT7_API_KEY = "<token>"       # PowerShell
```

## Cloudflare MCP Servers (cloudflare/mcp-server-cloudflare)

Path: `mcp_tools/mcp-server-cloudflare`

Cloudflare publishes a suite of remote MCP servers that surface services such
as Workers bindings, observability, Radar analytics, documentation search, and
more. We vendor the upstream monorepo so we always have the latest manifest
and development tooling checked into our workspace.

Initial setup (already executed):

```bash
cd mcp_tools/mcp-server-cloudflare
corepack pnpm install
```

> **Note**
> The repo contains multiple Cloudflare Workers projects. Running them locally
> requires a Cloudflare account, `wrangler` login, and service-specific API
> tokens. For most day-to-day work we simply proxy the hosted servers instead
> of standing them up ourselves.

### Using the hosted Cloudflare endpoints

All of the servers expose SSE endpoints (see `README.md` inside the repo for
the full list). You can bridge one into any MCP client via
[`mcp-remote`](https://www.npmjs.com/package/mcp-remote). Example command we
used to validate connectivity to the documentation server:

```bash
npx --yes mcp-remote https://docs.mcp.cloudflare.com/sse --print-manifest
```

You will be prompted for Cloudflare API tokens when a server requires them.
Refer to the upstream `README.md` for the exact scopes per service (for
instance the browser rendering server needs `Workers KV Storage:Read`,
`Workers R2 Storage:Read`, and related permissions).

## Integrating with MCP clients

Add the relevant command to any MCP-aware tool (Claude Desktop, Cursor,
VS Code extensions, etc.). Example configuration snippet:

```json
{
  "mcpServers": {
    "kubernetes": {
      "command": "npx",
      "args": ["--yes", "mcp-server-kubernetes"]
    },
    "context7": {
      "command": "npx",
      "args": ["--yes", "@upstash/context7-mcp"]
    },
    "cloudflare-docs": {
      "command": "npx",
      "args": [
        "--yes",
        "mcp-remote",
        "https://docs.mcp.cloudflare.com/sse"
      ],
      "description": "Cloudflare documentation search via remote MCP"
    }
  }
}
```

The stdio commands above also work with `mcp-chat` or the
Model Context Protocol Inspector if you want to sanity-check tool discovery
before wiring them into an editor.
