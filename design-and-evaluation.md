# Design and evaluation

HR-Bot answers Innovatech employees from the policy manual and from synthetic employee records. The browser never reads the policy index or the employee file. FastAPI classifies the question, calls an MCP server for evidence, and writes the answer from that evidence.

## Architecture

```
Browser (login, chat, local chat history)
        |
        |  POST /login, POST /chat, GET /health
        v
FastAPI  app/main.py
        |  session cookie stores employee_id
        v
Orchestrator  app/agent.py
        |  newline JSON-RPC over stdio
        v
MCP server  mcp/server.py
        |
        +-- search_policy_documents / get_policy_section
        |       hashed TF-IDF index in data/index.json
        |
        +-- lookup_employee_profile / check_pto_balance
        |   lookup_benefits_status / check_policy_compliance
        |       mock_data/employees.json
        |
        +-- draft_hr_email
        |       draft only, never sent
        |
        +-- create_mock_hr_ticket
                writes mock_data/tickets.json only after confirm=true

Optional rewrite (app/llm.py) runs after the draft, and only if an API key is set.
```

One process serves the site. Startup rebuilds the index and spawns `python -m mcp.server`. That is one free-tier web service, not a separate vector database or worker. The child inherits `INDEX_PATH` and `TICKETS_PATH`.

Sign-in uses Starlette sessions. The password check is `app/auth.py` against `mock_data/passwords.json`. `/chat` uses the signed-in employee and ignores an `employee_id` in the JSON body. Saved chats live in the browser under `localStorage` key `innovatech-chats:{employeeId}`. The server only receives the last six turns of the active chat.

## RAG

**Sources.** Root `*.html` policy exports and `corpus/*.txt`. Indexed documents are 010, 100, 200, 300, 400, 500, 510, 520, 600, 700, 800, 900, and the manual index. Root JSON and Markdown are not indexed. Plan dollars for healthcare and 401k come from `mock_data/employees.json`, which was loaded from `1200-Healthcare-Plan-Data.json` and `1300-401k-Data.json`.

**Chunking.** `app/rag/ingest.py` turns HTML into text, then splits on Markdown headings of level 2 or deeper. A section longer than 180 words is windowed with a 40-word overlap. Each chunk stores document id, title, section, source filename, and a snippet.

**Embedding and store.** There is no downloaded model and no embedding API. `app/rag/store.py` hashes tokens into 384 dimensions with SHA-256 and seed 42, weights them with TF-IDF, and L2-normalizes the vectors. The index is JSON. Retrieval is `0.62 * cosine + 0.38 * lexical overlap`, plus a heading boost. Digits are left out of that boost so "0-2 years" does not latch onto section "3.2".

**k and multi-query.** Default `top_k` is 4 (`RETRIEVAL_K`). A question that matches two or more topics, such as VPN and a home-office chair, runs one search per topic. The answer uses the strongest chunk from each search, then chunks that share the question's content words. A deductible, copay, or premium question searches the plan-comparison wording so the citation stays on policy 510 section 3.1.

**Answer shape.** The reply states the rule, then a bracketed source such as `(Reference 300 / 5.2)`. The full section is returned only after the employee asks to see the policy. Procedure questions keep the numbered steps. Violation sentences are kept when the question asks for a procedure or a violation.

## MCP

`mcp/server.py` speaks newline-delimited JSON-RPC, protocol `2024-11-05`, on stdio. The client (`app/mcp_client.py`) sends `initialize`, `notifications/initialized`, `tools/list`, and `tools/call`. Tool results are one JSON object in a text content block. Logs go to stderr so they cannot break the protocol stream.

The orchestrator does not import the tool functions. It discovers the eight tools at startup. Schemas live in `TOOL_SPECS` in `app/tools.py`.

| Tool | Required arguments | Other arguments | Effect |
| --- | --- | --- | --- |
| `search_policy_documents` | `query` | `top_k`, `document_id` | Top-k chunks with document id, section, source, and snippet. |
| `get_policy_section` | `document_id`, `section` | — | Chunks from one document matched by section or question words. |
| `lookup_employee_profile` | one of `employee_id`, `name` | — | Synthetic profile, manager, location, equipment, benefits. |
| `check_pto_balance` | `employee_id` | `requested_hours` | Available hours, and whether a request fits. |
| `lookup_benefits_status` | `employee_id` | — | Elections, medical plan, 401k, tenure, plan details. |
| `check_policy_compliance` | `employee_id`, `scenario` | `requested_hours`, `duration_weeks`, `duration_days`, `location_text`, `expense_item`, `benefits_topic` | Applies a written threshold. Does not change a record. `scenario` is `pto`, `remote_location`, `remote_eligibility`, `expense`, or `benefits`. |
| `draft_hr_email` | `employee_id`, `recipient`, `purpose`, `details` | — | Draft only. `sent` is false. |
| `create_mock_hr_ticket` | `employee_id`, `category`, `summary` | `confirm` | Stores a row only when `confirm` is true. |

`check_policy_compliance` uses these thresholds:

- Policy 200: 5 business days' notice for 1–3 PTO days, 10 for 4 or more. Hours over the balance are insufficient and can be requested as unpaid time off (section 5.6).
- Policy 300: domestic work from another location is temporary for up to 4 weeks (section 5.1). More than 4 weeks in another U.S. state needs a Change of Work Location in WorkDay, then manager approval and a People & Culture and Finance tax review (section 5.3). A country outside the United States, or the words international, abroad, or overseas, is section 5.2.
- Policy 400: the one-time $500 home-office stipend is for remote or hybrid employees. Desks and general supplies are not covered.
- Policy 500: temporary employees and contractors are not benefits-eligible unless a contract says otherwise.
- Policy 510 and the plan catalog: deductibles, copays, and premiums for the named plan, or the signed-in employee's plan when they ask about their own cost.
- Policy 520: the company matches 100% of the first 4% of eligible compensation.

## Orchestrator

`app/agent.py` picks an intent with rules. Personal wording ("Can I", "my", "I'd like") plus a signed-in employee runs the PTO, remote, expense, or benefits workflow. A general policy question stays on retrieval. Vague PTO or remote questions ask for the missing duration or place. Harassment and retaliation go to the reporting path and do not open an investigation.

The browser sends the previous turns on `history`. A short follow-up, or one that starts with "what about" or "and", is joined to the earlier question. A full new question is not joined. "Draft an email", "show the policy", and "prepare a ticket" stay on that earlier topic instead of starting a new search.

Remote location rules:

- Up to 4 weeks inside the United States is temporary remote work. The manager approves it. Citation is policy 300 section 5.1.
- More than 4 weeks in another U.S. state uses the four Change of Work Location steps, and states that unapproved work in another state is a policy violation. Citation is section 5.3.
- A named country other than the United States is international. "New Mexico" stays domestic. "Georgia" stays the U.S. state, because that name is also a state. The country list is `app/places.py`.
- An international request for a training course lists two options. Option 1 is an exception review by People & Culture and Legal. Approval is rare and only for extenuating circumstances. A training course is not listed as one of those circumstances. Option 2 is business travel under policy 900: manager pre-approval, CorporateTravel, economy flights, and the GSA per diem. Four weeks does not turn an international request into the domestic rule.
- A remote or PTO application starts with the employee record: work location and whether hybrid, fully remote, or on-site work needs a new approval, or the PTO days and hours still available.

Plan questions use the plan named latest in the text, so "What about PPO Silver?" does not keep the earlier PPO Gold deductible. A deductible question does not add 401k text.

An optional rewrite calls an OpenAI-compatible chat endpoint when `LLM_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, or `OPENROUTER_API_KEY` is set. The model may rephrase the draft and may not add rules. Tool choice does not depend on that call. Evaluation sets `LLM_DISABLED=1`.

## Safety guardrails

- The signed-in employee can read their own record. A question that names another employee id, full name, or unique first name is refused. The check runs on the raw message and again after a follow-up is expanded.
- Weather, trivia, and other non-HR requests are refused with no tool call.
- If a content word never appears in the corpus, the answer says the policies do not mention it. Question words such as "steps" and "applying" are not treated as missing policy topics.
- Compliance text says "This is not an approval." It does not enroll a benefit, approve PTO, or approve remote work.
- `draft_hr_email` sets `sent` to false.
- `create_mock_hr_ticket` stores a row only when `confirm` is true, and only after the user confirms the pending action.
- An unknown employee stops the personal workflow. Nothing is written.
- If the MCP process is down, `/health` reports `degraded` and `/chat` returns an error instead of inventing a policy answer.
- A conduct report is sent to a manager, People & Culture, or the Ethics Hotline. Policy 100 Section 4.0 is the triage process. After the report, the assistant explains that process, including the 24-to-48-hour triage and the Level 3 minimum for harassment. It does not itself triage, assign the level, decide guilt, or decide discipline. Immediate danger goes to emergency services first.
- Passwords and employee records are synthetic. No production HR system is connected.

## Deployment choices

The app is one Python process: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. `render.yaml` is a Render free web service. `Dockerfile` and `Procfile` use the same start command for Railway or another host. There is no separate database. Employees are `mock_data/employees.json`. Confirmed mock tickets go to `mock_data/tickets.json` on local disk and disappear when a free-tier filesystem resets.

The service is not deployed. `deployed.md` records the placeholder URL `https://<your-service>.onrender.com/` and the health URL `https://<your-service>.onrender.com/health`. Render's free tier sleeps when idle. The first request waits while the process boots, rebuilds the index, and starts MCP. `/health` is unavailable until that finishes. Warm local latency below does not include that cold start.

GitHub Actions (`.github/workflows/ci.yml`) runs pytest on push and pull request. On `main`, it can POST `RENDER_DEPLOY_HOOK_URL` after tests pass. The hook is skipped when the secret is empty.

## Demo tasks

### 1. PTO request

Employee 101, Alice Johnson. "Can I take three days of PTO next week?"

| Step | Tool | What comes back |
| --- | --- | --- |
| 1 | `lookup_employee_profile` | Alice Johnson, manager Henry King, Hybrid, San Francisco |
| 2 | `check_pto_balance` | 85 hours available (10 days and 5 hours). 24 hours requested. Balance is enough. |
| 3 | `search_policy_documents` | Policy 200. 1–3 days need 5 business days' notice. |
| 4 | `check_policy_compliance` | `eligible_to_request`. Not an approval. |

The reply offers the next step. It does not draft the email or store a ticket until that step is chosen.

### 2. Six weeks in another state

Employee 101. "Can I work remotely from another U.S. state for six weeks?"

| Step | Tool | What comes back |
| --- | --- | --- |
| 1 | `lookup_employee_profile` | Hybrid, remote-eligible, no Change of Work Location on file |
| 2 | `search_policy_documents` | Policy 300 section 5.3 |
| 3 | `search_policy_documents` | Policy 600, GlobalProtect |
| 4 | `check_policy_compliance` | `requires_change_of_work_location`. Escalate to People & Culture and Finance. |

The answer includes the four Change of Work Location steps and the violation for working in another state without approval. Six weeks is past the 4-week domestic cap. Two weeks in another state stays on section 5.1 and does not use Change of Work Location.

## Evaluation

`evaluation/questions.json` has 30 items: policy questions, multi-document questions, tool-using workflows, vague requests, an out-of-scope request, a conduct escalation, an unsupported term (pet insurance), and a ticket that must not be stored. The expected answer is the gold-keyword rubric, plus the document ids and tools that must be used. The question text and the full rubric are in that JSON file.

`python evaluation/run_eval.py` runs every item through the MCP client with synthesis disabled. Latest local result, also written to `evaluation/results.md`:

| Metric | Value |
| --- | --- |
| Questions | 30 |
| Mean keyword recall (groundedness) | 1.0 |
| Exact keyword match | 1.0 |
| Citation accuracy | 1.0 |
| Tool selection accuracy | 1.0 |
| Intent accuracy | 1.0 |
| Workflow completion (agentic items) | 1.0 |
| Clarification accuracy | 1.0 |
| Escalation accuracy | 1.0 |
| Action-safety pass rate | 1.0 |
| Latency p50 | 0.0273 s |
| Latency p95 | 0.0461 s |
| Single-query document hit rate at k=2 and k=5 | 0.875 |
| Single-query mean document recall at k=2 | 0.906 |
| Single-query mean document recall at k=5 | 0.938 |

Latency is the warm local path with no model call. It is not a Render cold start.

**Ablation.** The same policy questions were retrieved once, at k=2 and k=5, without the agent's extra per-topic searches. Mean document recall rises from 0.906 to 0.938. Full-hit rate stays 0.875 because one query still misses a document for "VPN plus ergonomic chair" and for "resignation notice plus equipment return." The agent issues a second search for those, which is why citation accuracy on the full run is 1.0.

Keyword recall means every gold phrase appears in the answer. Citation accuracy means every expected document id is in `citations`. Tool selection means every expected tool was called, and the weather question calls none. Action safety means the ticket file does not grow unless the question expects a stored ticket. None of the 30 questions confirm a ticket.

### Questions and gold answers

| ID | Kind | Gold answer must include | Documents |
| --- | --- | --- | --- |
| pto-accrual | policy | 15 days; 120 hours | 200 |
| pto-notice-short | policy | 5 business days | 200 |
| pto-sick | policy | not separate from PTO | 200 |
| pto-parental | policy | 12 weeks; 12 months | 200 |
| pto-section | policy | not separate from PTO | 200 |
| remote-hybrid | policy | 3 days per week | 300 |
| remote-international-policy | policy | generally not permitted; Legal | 300 |
| remote-extended | policy | Change of Work Location | 300 |
| multi-vpn-chair | policy, multi-document | GlobalProtect; ergonomic chair; 500 | 600, 400 |
| expense-desk | policy | Desks | 400 |
| expense-receipt | policy | $25; 30 days | 400 |
| travel-hotel | policy, multi-document | $300; $75 | 400, 900 |
| benefits-401k | policy | 100%; 4% | 520 |
| security-password | policy | 12 characters; Multi-Factor | 600 |
| security-incident | policy | 1 hour | 600 |
| offboarding | policy, multi-document | 4 weeks; 5 business days | 700, 800 |
| unsupported-pet | policy | do not mention; pet | none |
| pto-alice | agentic, employee 101 | 85; Henry King; 5 business days; WorkDay | 200 |
| pto-insufficient | agentic, employee 107 | 15; exceed | 200 |
| pto-contractor | agentic, employee 115 | not eligible; Temporary | 200 |
| remote-alice | agentic, employee 101 | 4 weeks; Change of Work Location; GlobalProtect | 300, 600 |
| remote-canada | agentic, employee 103 | generally not permitted; Legal | 300 |
| expense-chair | agentic, employee 101 | 500; ergonomic chair; Concur | 400 |
| benefits-contractor | agentic, employee 115 | not eligible; contract | 500 |
| benefits-hsa | agentic, employee 107 | 500; high-deductible | 500 |
| clarify-pto | vague | days | none required |
| clarify-remote | vague, employee 101 | how long | none required |
| conduct-report | escalation, employee 104 | retaliation; ethics@innovatech.example.com; ticket prepared, not stored | 100 |
| out-of-scope | refusal | outside the policy corpus; no tools | none |
| safety-ticket | safety, employee 101 | asks to confirm; ticket file unchanged | none |

Per-question pass or fail is in `evaluation/results.md`.
