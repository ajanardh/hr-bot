# Deployment

The application is not deployed yet. This workspace does not have a Render or Railway login, so there is no public URL to record.

## Intended service

| Item | Value |
| --- | --- |
| Host | Render free web service, or Railway using the same start command |
| Blueprint | `render.yaml` |
| Start | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| App URL | `https://<your-service>.onrender.com/` |
| Health URL | `https://<your-service>.onrender.com/health` |
| Chat API | `POST https://<your-service>.onrender.com/chat` |

After you deploy, replace the placeholders above with the real URL.

## Cold start

Render's free web services sleep after a period with no traffic. The first request then waits while the service boots. Startup rebuilds the local policy index (a small JSON file of hashed TF-IDF vectors) and spawns the MCP server process. `/health` stays unavailable until that finishes, then returns `status: ok` and `mcp: connected`.

Warm requests on this machine, measured by `evaluation/run_eval.py` with LLM synthesis off, are about 0.008 seconds at p50 and 0.014 seconds at p95. Those numbers do not include a host cold start. A sleeping free-tier service commonly takes several tens of seconds to a minute to wake. Run `/health` once before the demo if the tab has been idle.

No paid database is required. Employees live in `mock_data/employees.json`. Mock tickets, if confirmed, are written to `mock_data/tickets.json` on the service disk and disappear when the free-tier filesystem is reset.

## Environment

Do not commit API keys. Set `LLM_API_KEY` on the host only if you want the optional rewrite step. The demo tasks work with that variable unset.
