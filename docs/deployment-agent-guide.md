# MTP Deploy Agent — Setup & Usage Guide

An MCP server that deploys the MTP Flight Delay app to Render + Vercel directly from Claude Code.

---

## Prerequisites

- Python 3.11+ installed
- Node.js 18+ installed (for the Vercel frontend build)
- Your project pushed to a public GitHub repo
- A Render account (free): render.com
- A Vercel account (free): vercel.com

---

## One-Time Setup

### 1. Install the agent's dependencies

```bash
cd MTP/deploy_agent
pip install -r requirements.txt
```

### 2. Get your API tokens

**Render API Key**
1. Go to dashboard.render.com
2. Click your avatar (top right) → Account Settings → API Keys
3. Click "Create API Key" → copy the value (starts with `rnd_`)

**Vercel Token**
1. Go to vercel.com → your avatar → Settings → Tokens
2. Click "Create" → name it "claude-deploy" → copy the value

### 3. Create `deploy_agent/.env`

```
RENDER_API_KEY=rnd_xxxxxxxxxxxx
VERCEL_TOKEN=xxxxxxxxxxxx
RENDER_SERVICE_ID=          ← fill after first deploy
VERCEL_PROJECT_ID=          ← fill after first deploy
```

### 4. Register with Claude Code

Open `~/.claude/settings.json` and add:

```json
{
  "mcpServers": {
    "deploy-agent": {
      "command": "python",
      "args": ["C:/Users/hp/OneDrive/Desktop/MTP/deploy_agent/server.py"],
      "env": {
        "RENDER_API_KEY": "rnd_xxxxxxxxxxxx",
        "VERCEL_TOKEN": "xxxxxxxxxxxx",
        "RENDER_SERVICE_ID": "",
        "VERCEL_PROJECT_ID": ""
      }
    }
  }
}
```

### 5. Restart Claude Code

Close and reopen Claude Code. Run `/mcp` to confirm `deploy-agent` appears.

---

## Usage

### Deploy the full stack

```
Deploy the full stack to production using github repo 21MA23002/mtp-flight-delay
```

Claude will:
1. Create the Render web service and wait for it to go live (~3–5 min)
2. Create the Vercel project, inject `VITE_API_URL` automatically
3. Deploy the frontend (~1–2 min)
4. Return both URLs

Expected output:
```json
{
  "status": "success",
  "backend_url": "https://mtp-flight-api.onrender.com",
  "frontend_url": "https://mtp-flight-delay.vercel.app",
  "render_service_id": "srv-xxxx",
  "vercel_project_id": "prj-xxxx"
}
```

**After the first deploy**, save the returned IDs to `deploy_agent/.env` and `settings.json`:
```
RENDER_SERVICE_ID=srv-xxxx
VERCEL_PROJECT_ID=prj-xxxx
```

### Check backend status

```
Is the backend still up?
```

### View backend logs

```
Show me the backend logs
```

### Check frontend status

```
What's the status of the frontend deployment?
```

---

## Re-deploying After Code Changes

After pushing new code to GitHub main branch:

```
Re-deploy the full stack
```

The agent calls `render.trigger_deploy` on the existing service (no recreation) and `vercel.trigger_via_hook` on the existing project.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Deploy agent misconfigured: RENDER_API_KEY not found" | Add token to `deploy_agent/.env` and/or `settings.json` |
| Render deploy times out | Free tier has cold starts. Check logs: "Show me the backend logs" |
| ModuleNotFoundError on startup | Run `pip install -r deploy_agent/requirements.txt` |
| Vercel build fails | Check that `frontend/` directory exists and has a valid `package.json` |
| CORS errors on the live site | The CORS middleware in `api/main.py` allows `*` — no change needed |

---

## Free Tier Notes

- **Render free tier**: spins down after 15 min of inactivity. First request after sleep takes ~30s. Upgrade to the $7/month Starter plan for always-on.
- **Vercel free tier**: unlimited deployments, 100 GB bandwidth/month. Sufficient for a thesis demo.
- **GitHub**: model `.pkl` files must be under 100 MB each. Use `git lfs track "*.pkl"` if any exceed the limit.
