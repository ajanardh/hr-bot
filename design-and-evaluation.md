# Design and evaluation

Innovatech HR Assistant answers employee questions from the policy manual and from synthetic HR records. The web app never calls the record store or the vector index itself. It asks an MCP server to do that, then writes an answer from the tool output.

## Architecture

```
Browser  -- POST /chat -->  FastAPI (app/main.py)
                               |
                               v
                         Orchestrator (app/agent.py)
                               |
                               |  newline JSON-RPC over stdio
                               v
                         MCP server (mcp/server.py)
                          /            \
                         /              \
            search / get_section     employee, PTO, benefits,
            local vector index       compliance, draft email,
            data/index.json          confirm-to-store ticket
                         \              /
                          v            v
                    Optional LLM rewrite
                    (only if LLM_API_KEY is set)
```

One process serves the site. On startup it rebuilds the index and spawns `python -m mcp.server` as a child process. That keeps a free-tier deploy to a single Render or Railway service. The child inherits `INDEX_PATH` and `TICKETS_PATH`.

## RAG

**Sources.** HTML exports in the project root (policies 100–900 and the index) and plain text in `corpus/`. The ingester also accepts Markdown. HTML is converted by stripping tags. Both formats are chunked the same way after that.

**Chunking.** Sections split on Markdown headings (`##` and deeper). A section longer than 180 words is windowed with a 40-word overlap. Each chunk stores document id, title, section, source filename, and a snippet. The seed for the embedder is 42, so the same corpus always produces the same vectors.

**Embedding and store.** There is no downloaded model and no embedding API. Tokens are hashed into 384 dimensions with SHA-256 and seed 42, weighted with TF-IDF, and L2-normalized. Vectors live in a JSON file. Retrieval mixes cosine similarity (0.62) with lexical overlap (0.38), then adds a heading boost. Numeric tokens are ignored in that boost so a query like "0-2 years" does not latch onto section "3.2".

**k and multi-query.** Default `top_k` is 4 (`RETRIEVAL_K`). A question that matches two or more topics (for example VPN and a home-office chair) triggers one search per topic. The answer quotes the top chunk from each search, then any chunk that shares more of the question's words.

**Citations.** A direct answer states the rule and ends with a bracketed source, such as `(Reference 200 / 3.1)`. The full section is returned only when the employee chooses "Show the policy." The API still returns snippet metadata on `citations`.

**Guardrails.**

- Weather, trivia, and other non-HR requests are refused without a tool call.
- If a content word never appears in the corpus ("pet"), the answer says the documents do not mention it instead of treating a nearby benefits paragraph as a pet-insurance rule.
- Policy facts are the quoted sections. The compliance decision is labeled as an assessment, not an approval.
- Draft emails set `sent: false`.
- `create_mock_hr_ticket` stores a row only when `confirm` is true. The orchestrator sets that flag only after the user confirms the pending action.
- A missing or unknown employee id stops the personal workflow and asks for a correction. Nothing is written.
- If the MCP process is down, `/health` reports `degraded` and `/chat` returns an error trace instead of inventing a policy answer.

## MCP

Transport is stdio, newline-delimited JSON-RPC, protocol version `2024-11-05`. The client sends `initialize`, `notifications/initialized`, `tools/list`, and `tools/call`. Tool results are a JSON object in a text content block. Logs go to stderr so they cannot corrupt the protocol stream.

The client discovers tools at startup with `tools/list`. These eight tools are registered:

| Tool | Role |
| --- | --- |
| `search_policy_documents` | Top-k policy retrieval. Optional `document_id` filter. |
| `get_policy_section` | One document, matched by section or question words. |
| `lookup_employee_profile` | Synthetic profile, manager, equipment, location requests. |
| `check_pto_balance` | Available hours versus a requested amount. |
| `lookup_benefits_status` | Elections, plan, tenure. |
| `check_policy_compliance` | Applies policy thresholds to one employee. Scenarios: `pto`, `remote_location`, `remote_eligibility`, `expense`, `benefits`. Does not change records. |
| `draft_hr_email` | Draft only. Never sends. |
| `create_mock_hr_ticket` | Returns a draft unless `confirm` is true. |

Thresholds inside `check_policy_compliance` follow the manual: PTO notice of 5 business days for 1–3 days and 10 for 4 or more (Policy 200); 4-week domestic cap and the Change of Work Location path (Policy 300); $500 stipend and ineligible desks (Policy 400); temporary and contract staff excluded from benefits (Policy 500). The answer still quotes the retrieved section so the number is tied to a source, not only to the tool.

## Orchestrator

Intent is chosen with rules, not a hidden chain of thought. The trace the UI shows is the operational one: tool name, arguments, status, and JSON output.

Personal workflows run only when the wording is personal ("Can I…", "employee 115") or an employee id is selected. A general "how many days" question stays on retrieval. Vague PTO or remote questions ask for the missing days or location and still quote the relevant policy. Harassment and retaliation route to the reporting path in Policy 100 and the case-handling procedure, recommend escalation, and prepare a ticket without storing it.

An optional rewrite calls an OpenAI-compatible chat endpoint when `LLM_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, or `OPENROUTER_API_KEY` is set. The model is told to rephrase the evidence draft and not add rules. Tool choice does not depend on that call. Evaluation turns the rewrite off with `LLM_DISABLED=1`.

## Demo tasks

### 1. PTO request

Employee **101**, Alice Johnson. Message: "Can I take three days of PTO next week?"

| Step | Tool | Arguments that matter | What comes back |
| --- | --- | --- | --- |
| 1 | `lookup_employee_profile` | `employee_id=101` | Alice Johnson, manager Henry King, hybrid, San Francisco |
| 2 | `check_pto_balance` | `requested_hours=24` | 85 hours available, balance is enough |
| 3 | `search_policy_documents` | PTO notice / WorkDay query | Policy 200, including the 5-business-day notice for 1–3 days |
| 4 | `check_policy_compliance` | `scenario=pto`, `requested_hours=24` | `eligible_to_request`, not an approval |
| — | Next-step options | Not called until chosen | Draft an email, show the policy, or prepare a mock ticket |

Three days is 24 hours. 85 is greater than 24. Policy 200 still requires manager approval in WorkDay and at least 5 business days' notice. No ticket is created.

### 2. Six weeks in another state

Employee **101**. Message: "Can I work remotely from another U.S. state for six weeks?"

| Step | Tool | Arguments that matter | What comes back |
| --- | --- | --- | --- |
| 1 | `lookup_employee_profile` | `employee_id=101` | Hybrid, remote-eligible, no change-of-location on file, international remote not approved |
| 2 | `search_policy_documents` | temporary / extended remote location | Policy 300 sections 5.1 and 5.3 |
| 3 | `search_policy_documents` | VPN and device security | Policy 600, GlobalProtect |
| 4 | `check_policy_compliance` | `scenario=remote_location`, `duration_weeks=6`, the user text as `location_text` | `requires_change_of_work_location`; escalate to People & Culture and Finance |
| — | Next-step options | Not called until chosen | Draft an email, show the policies, or prepare a mock ticket |

Six weeks is past the 4-week temporary domestic cap, so the extended path applies: WorkDay Change of Work Location, manager approval, then a tax review. The employee record's 20-day temporary counter is the same cap counted in business days, and it is also exceeded. Working this way is not approved by the assistant.

Both buttons on the home page send these requests. `GET /api/demo-tasks` returns the same payloads for a grader using curl.

## Evaluation

`evaluation/questions.json` has 30 items: policy questions, one multi-document question, tool-using workflows, vague requests, an out-of-scope request, a conduct escalation, an unsupported term (pet insurance), and a ticket that must not be stored. Gold answers are the keywords that have to appear, plus the document ids and tools that have to be used.

`python evaluation/run_eval.py` runs every item through the MCP client with synthesis disabled. Latest local result:

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
| Latency p50 | 0.0079 s |
| Latency p95 | 0.0136 s |
| Single-query recall at k=2 | 0.906 |
| Single-query recall at k=5 | 0.938 |
| Single-query full-hit at k=2 and k=5 | 0.875 |

Latency is the warm local path with no model call. It is not a cold start on Render. See `deployed.md`.

**Ablation.** The same policy questions were retrieved once, at k=2 and k=5, without the agent's extra per-topic searches. Raising k from 2 to 5 lifts mean document recall from 0.906 to 0.938. Full-hit rate stays 0.875: a single query still fails to surface every document for "VPN plus ergonomic chair" and for "resignation notice plus equipment return." Those two need the second search the agent already issues, which is why citation accuracy on the full run is 1.0 while the single-query ablation is not.

Keyword recall is the groundedness score used here: the gold phrase has to show up in the answer, and the answer is built from snippets and tool fields. Citation accuracy requires every expected document id to be in `citations`. Tool selection requires every expected tool to have been called (and, for the weather question, no tools at all). Action safety requires that the ticket file does not grow unless the question says a ticket should be created. None of the 30 questions confirm a ticket.

### Questions and gold answers

The message text is the question. Gold keywords are the expected answer rubric. Full JSON, including tool lists, is `evaluation/questions.json`.

| ID | Kind | Gold answer must include | Documents |
| --- | --- | --- | --- |
| pto-accrual | policy | 15 days; 120 hours | 200 |
| pto-notice-short | policy | 5 business days | 200 |
| pto-sick | policy | not separate from PTO | 200 |
| pto-parental | policy | 12 weeks; 12 months | 200 |
| pto-section | policy | not separate from PTO (via `get_policy_section` and search) | 200 |
| remote-hybrid | policy | 3 days per week | 300 |
| remote-international-policy | policy | generally not permitted; Legal | 300 |
| remote-extended | policy | Change of Work Location | 300 |
| multi-vpn-chair | policy, multi-document | GlobalProtect; ergonomic chair; 500 | 600, 400 |
| expense-desk | policy | Desks | 400 |
| expense-receipt | policy | $25; 30 days | 400 |
| travel-hotel | policy, multi-document | $300; $75 | 400, 900 |
| benefits-401k | policy | 100%; 4% | 500 |
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
| clarify-pto | vague | asks for the number of days | none required |
| clarify-remote | vague, employee 101 | asks how long | none required |
| conduct-report | escalation, employee 104 | retaliation; ethics@innovatech.example.com; ticket prepared, not stored | 100 |
| out-of-scope | refusal | outside the policy corpus; no tools | none |
| safety-ticket | safety, employee 101 | asks to confirm; ticket file unchanged | none |

Per-question pass/fail detail is in `evaluation/results.md`.
