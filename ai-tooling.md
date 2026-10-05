# AI tooling

This application was built in Cursor with the Grok coding agent. The agent read the assignment PDF, the HTML policy exports, and the RTF-wrapped employee file, then generated the web app, retrieval index, MCP server, tests, and evaluation set.

## What worked well

- Generating the project skeleton from the written requirements: FastAPI chat UI, a stdio MCP process, and a separate orchestrator that is not allowed to import the tools directly.
- Turning the messy source files into something the index can read. The policies are HTML, not plain text, and the employee file is RTF around JSON. The agent extracted both and kept the originals.
- Using the evaluation script as a feedback loop. The first run missed accrual numbers, the VPN excerpt, and the security-incident question. Those failures were specific enough to fix the ranker and the intent rules, and the rerun scored the gold keywords.

## What did not

- The first retrieval ranker treated the digit in "0-2 years" as a match for section numbers such as "3.2". The accrual table was retrieved but fell out of the top four, so the answer cited Policy 200 and still omitted "15 days" and "120 hours".
- A single search is a weak way to answer a question that spans VPN rules and the home-office stipend. The agent needed one search per topic. The ablation in `evaluation/results.md` records that gap.
- A hosted chat model was not required to finish the workflows. Leaving synthesis optional avoids inventing policy numbers when no API key is configured. If a key is set, the model is only allowed to rephrase the draft built from tool output.

The tests in `tests/` and `python evaluation/run_eval.py` are the check on that generated code. They do not replace reading the cited policy section when a number matters.
