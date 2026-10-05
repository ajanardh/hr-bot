# Innovatech HR Assistant

Agentic HR assistant for Innovatech Solutions. It answers from the company policy manual, looks up synthetic employee records, and calls those capabilities through an MCP server. Policy answers cite the document and section they came from. Creating a mock HR ticket requires an explicit confirmation.

Two workflows are wired through the UI as demo buttons:

1. **PTO request** for employee 101: three days next week.
2. **Remote work** for employee 101: six weeks from another U.S. state.

## What is in this folder

The policy manual was provided as HTML exports (a markdown manual saved by a rich-text editor). Employee records were provided as `Mock-Employee-Data.json`, which is JSON wrapped in RTF. The app reads a cleaned copy at `mock_data/employees.json`. `corpus/010-hr-case-handling.txt` is an added plain-text procedure so ingestion covers both HTML and text.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` only if you want a hosted model to rewrite the final wording. With no key, the assistant still answers from retrieved policy text and tool results.

## Run locally

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The index is rebuilt on startup. The process then starts `python -m mcp.server` over stdio.

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

The home page asks for an employee id and password before the chat is shown. Passwords are in `mock_data/passwords.json`, one per employee. A signed-in employee can ask about their own record and about general policy. Requests about another employee are refused. A body field named `employee_id` on `/chat` is ignored.

Reproduce the two demo tasks as Alice Johnson (employee 101). Read her password from the JSON file, then keep the cookie:

```bash
curl -s -c /tmp/innovatech.cookies http://127.0.0.1:8000/login \
  -H 'Content-Type: application/json' \
  -d '{"employee_id":"101","password":"<password from mock_data/passwords.json>"}'

curl -s -b /tmp/innovatech.cookies http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Can I take three days of PTO next week?"}'

curl -s -b /tmp/innovatech.cookies http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Can I work remotely from another U.S. state for six weeks?"}'
```

Each response includes `answer`, `citations`, `snippets`, and `trace` (tool name, arguments, and output).

## Tests and evaluation

```bash
pytest -q
python evaluation/run_eval.py
```

`evaluation/questions.json` holds 30 questions with gold keywords, expected document ids, and expected tools. `evaluation/results.md` is the latest local report. The run disables LLM synthesis so the scores stay repeatable.

## Deploy

`render.yaml` is set up for a single free Render web service. `Dockerfile` and `Procfile` cover the same start command if you use Railway or another host.

1. Push this directory as its own GitHub repository. The git metadata currently detected above this folder belongs to `Downloads`, so create a new repo from `Innovatech_solutions` rather than committing that parent folder.
2. On Render, create a Blueprint from `render.yaml`, or a Python web service with build `pip install -r requirements.txt` and start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
3. Optional env vars: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`. Leave them empty to keep the extractive composer.
4. After the first successful GitHub Actions run on `main`, add repository secret `RENDER_DEPLOY_HOOK_URL` if you want Actions to trigger a deploy. The deploy job does not run unless tests pass, and it skips the hook when the secret is absent.

Put the live URL in `deployed.md`.

## Layout

| Path | Role |
| --- | --- |
| `app/main.py` | Chat UI, `POST /chat`, `GET /health` |
| `app/agent.py` | Orchestrator |
| `app/mcp_client.py` | Stdio MCP client |
| `mcp/server.py` | MCP server and tool schemas |
| `app/tools.py` | Tool implementations used only by the server |
| `app/rag/` | HTML and text ingestion, hashed TF-IDF index, retrieval |
| `mock_data/` | Synthetic employees and mock tickets |
| `evaluation/` | Questions, runner, results |
| `.github/workflows/ci.yml` | Test, then optional deploy hook |
