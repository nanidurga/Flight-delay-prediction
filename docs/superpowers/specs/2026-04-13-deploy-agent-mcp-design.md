# Deploy Agent MCP Server — Design Spec

**Date:** 2026-04-13
**Project:** MTP — Flight Delay Prediction (21MA23002)
**Goal:** Build a multi-module Python MCP server that exposes Render + Vercel deployment as tools inside Claude Code, enabling full-stack deployment via natural language.

---

## 1. What We Are Building

A Python MCP server (`deploy_agent/`) that registers four tools into Claude Code:

| Tool | Purpose |
|---|---|
| `deploy_full_stack` | End-to-end: deploys FastAPI backend to Render, waits for live URL, sets it as `VITE_API_URL` on Vercel, deploys frontend |
| `render_get_logs` | Fetches last 100 lines of Render runtime logs |
| `render_get_status` | Returns current deploy state + uptime for the Render service |
| `vercel_get_status` | Returns latest Vercel deployment state + live URL |

The primary workflow is `deploy_full_stack` — the three status/log tools exist for post-deploy debugging.

---

## 2. Folder Structure

```
MTP/
└── deploy_agent/
    ├── server.py          # MCP entry point — registers tools, starts stdio server
    ├── render.py          # Render REST API client
    ├── vercel.py          # Vercel REST API client
    ├── orchestrator.py    # deploy_full_stack() sequencing logic
    ├── config.py          # Token loading: .env first, env vars fallback
    ├── .env               # Local secrets (gitignored)
    └── requirements.txt   # mcp, httpx, python-dotenv
```

### Module responsibilities (strict — no cross-cutting)

- **`config.py`** — loads and validates tokens only; no HTTP
- **`render.py`** — one function per Render API operation; raises `RenderError` on failure
- **`vercel.py`** — one function per Vercel API operation; raises `VercelError` on failure
- **`orchestrator.py`** — sequences render + vercel calls; no direct HTTP
- **`server.py`** — MCP tool registration and stdio server start; no business logic

---

## 3. Tool Interface

### `deploy_full_stack`

**Input:**
```json
{
  "github_repo": "21MA23002/mtp-flight-delay",
  "service_name": "mtp-flight-api"  // optional, defaults to repo name
}
```

**Sequence:**
```
1. render.create_or_get_service(repo, name)
       POST /v1/services  (skips if service already exists)
       → service_id, deploy_id

2. render.wait_for_deploy(deploy_id)
       polls GET /v1/deploys/{id} every 10s, timeout 15 min
       → live_url  (e.g. https://mtp-flight-api.onrender.com)

3. vercel.create_or_get_project(repo, live_url)
       POST /v9/projects  (skips if project already exists)
       sets root_directory=frontend, framework=vite
       sets VITE_API_URL = live_url
       → project_id

4. vercel.trigger_deploy(project_id)
       POST /v13/deployments
       polls until state=READY, timeout 10 min
       → frontend_url  (e.g. https://mtp-flight-delay.vercel.app)
```

**Output:**
```json
{
  "backend_url": "https://mtp-flight-api.onrender.com",
  "frontend_url": "https://mtp-flight-delay.vercel.app",
  "render_service_id": "srv-xxxx",
  "vercel_project_id": "prj-xxxx",
  "status": "success"
}
```

### `render_get_logs`
**Input:** `{ "service_id": "srv-xxxx" }`
**Output:** `{ "lines": ["...", "..."] }` — last 100 log lines

### `render_get_status`
**Input:** `{ "service_id": "srv-xxxx" }`
**Output:** `{ "state": "live", "deploy_id": "dep-xxxx", "created_at": "..." }`

### `vercel_get_status`
**Input:** `{ "project_id": "prj-xxxx" }`
**Output:** `{ "state": "READY", "url": "https://...", "created_at": "..." }`

---

## 4. Token Configuration

### Priority order
1. `deploy_agent/.env` (primary — for local use)
2. `os.environ` — populated by Claude Code from `mcpServers.env` in `~/.claude/settings.json`
3. Raise `ConfigError` with clear instructions if token still missing

### Required tokens

| Variable | Where to get it |
|---|---|
| `RENDER_API_KEY` | render.com → Account Settings → API Keys |
| `VERCEL_TOKEN` | vercel.com → Settings → Tokens |
| `RENDER_SERVICE_ID` | Render dashboard URL: `dashboard.render.com/web/srv-xxxx` — only needed after first deploy |
| `VERCEL_PROJECT_ID` | Vercel project Settings → General → Project ID — only needed after first deploy |

`RENDER_SERVICE_ID` and `VERCEL_PROJECT_ID` are optional on first run — `deploy_full_stack` creates the resources and returns the IDs, which you then save for subsequent calls.

### `deploy_agent/.env` template
```
RENDER_API_KEY=rnd_xxxxxxxxxxxxxxxxxxxx
VERCEL_TOKEN=xxxxxxxxxxxxxxxxxxxx
RENDER_SERVICE_ID=srv-xxxx        # fill after first deploy
VERCEL_PROJECT_ID=prj-xxxx        # fill after first deploy
```

### `~/.claude/settings.json` snippet
```json
{
  "mcpServers": {
    "deploy-agent": {
      "command": "python",
      "args": ["C:/Users/hp/OneDrive/Desktop/MTP/deploy_agent/server.py"],
      "env": {
        "RENDER_API_KEY": "rnd_xxxx",
        "VERCEL_TOKEN": "xxxx",
        "RENDER_SERVICE_ID": "srv-xxxx",
        "VERCEL_PROJECT_ID": "prj-xxxx"
      }
    }
  }
}
```

---

## 5. Error Handling

| Scenario | Behaviour |
|---|---|
| Token missing | `ConfigError` raised at startup with message telling user which key is missing and where to add it |
| Render deploy fails | `RenderError` with deploy logs snippet attached |
| Render deploy times out (>15 min) | `TimeoutError` — returns partial result with `render_service_id` so user can call `render_get_logs` |
| Vercel deploy fails | `VercelError` with error message from API |
| Vercel project already exists | Silently reuses existing project (idempotent) |
| Render service already exists | Silently triggers new deploy on existing service |

---

## 6. Render Configuration for This Project

The Render service will be created with these settings (matching `CLOUD_DEPLOYMENT_GUIDE.html`):

```
runtime:       python
build_command: pip install -r requirements.txt
start_command: uvicorn api.main:app --host 0.0.0.0 --port $PORT
plan:          free
region:        oregon
env_vars:
  PYTHON_VERSION: "3.11.0"
```

`requirements.txt` must exist at the repo root (copy of `api/requirements.txt` + `lightgbm`). The orchestrator will check for it and raise a clear error if missing.

---

## 7. Vercel Configuration for This Project

```
root_directory: frontend
framework:      vite
build_command:  npm run build
output_dir:     dist
env_vars:
  VITE_API_URL: <injected from Render live_url>
```

---

## 8. End-to-End User Experience

```
You:     "Deploy the full stack to production"
Claude:  [calls deploy_full_stack(github_repo="21MA23002/mtp-flight-delay")]
         Step 1/4: Creating Render service...
         Step 2/4: Waiting for backend deploy (this takes 3-5 min)...
         Step 3/4: Backend live → https://mtp-flight-api.onrender.com
                   Creating Vercel project, injecting VITE_API_URL...
         Step 4/4: Deploying frontend...
         Done. Frontend: https://mtp-flight-delay.vercel.app

You:     "Show me the backend logs"
Claude:  [calls render_get_logs(service_id="srv-abc123")]

You:     "Is the backend still up?"
Claude:  [calls render_get_status(service_id="srv-abc123")]
         → {"state": "live", "deploy_id": "dep-xxx", "created_at": "..."}
```

---

## 9. Files to Create

| File | Action |
|---|---|
| `deploy_agent/config.py` | New |
| `deploy_agent/render.py` | New |
| `deploy_agent/vercel.py` | New |
| `deploy_agent/orchestrator.py` | New |
| `deploy_agent/server.py` | New |
| `deploy_agent/requirements.txt` | New |
| `deploy_agent/.env` | New (gitignored, template only) |
| `deploy_agent/.env.example` | New (committed, no real tokens) |
| `requirements.txt` | New at repo root (for Render build) |
| `.gitignore` | Update to include `deploy_agent/.env` |
| `docs/deployment-agent-guide.md` | New — complete user-facing documentation |

---

## 10. Out of Scope

- GitHub Actions retraining trigger (already handled by existing `retrain.yml`)
- AWS / GCP alternatives (documented in `CLOUD_DEPLOYMENT_GUIDE.html`, not automated)
- Custom domain setup
- Render paid plan upgrade automation
