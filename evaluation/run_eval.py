"""Score groundedness, citations, tool selection, safety, and latency.

The run is deterministic: LLM synthesis is disabled and the embedder seed is fixed.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["INDEX_PATH"] = os.environ.get("INDEX_PATH", str(Path(tempfile.mkdtemp()) / "index.json"))
os.environ["TICKETS_PATH"] = os.environ.get("TICKETS_PATH", str(Path(tempfile.mkdtemp()) / "tickets.json"))
os.environ["LLM_DISABLED"] = "1"

from app.agent import Agent  # noqa: E402
from app.data_store import list_tickets  # noqa: E402
from app.mcp_client import McpClient  # noqa: E402
from app.rag.store import build_index, search  # noqa: E402

QUESTIONS = ROOT / "evaluation" / "questions.json"
RESULTS_JSON = ROOT / "evaluation" / "results.json"
RESULTS_MD = ROOT / "evaluation" / "results.md"


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[index]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate() -> dict:
    questions = json.loads(QUESTIONS.read_text())
    build_index()
    client = McpClient()
    client.start()
    agent = Agent(client)
    rows = []
    try:
        for question in questions:
            before = len(list_tickets())
            started = time.perf_counter()
            result = agent.run(question["message"], employee_id=question.get("employee_id"))
            elapsed = time.perf_counter() - started
            answer = result["answer"].casefold()
            keywords = question.get("gold_keywords") or []
            hits = [keyword for keyword in keywords if keyword.casefold() in answer]
            called = [step.get("tool") for step in result["trace"] if step.get("tool")]
            cited = {item.get("document_id") for item in result["citations"]}
            expected_docs = question.get("expected_documents") or []
            expected_tools = question.get("expected_tools") or []
            keyword_recall = len(hits) / len(keywords) if keywords else 1.0
            citation_hit = all(doc in cited for doc in expected_docs) if expected_docs else None
            if question.get("exact_tools"):
                tool_hit = set(called) == set(expected_tools)
            else:
                tool_hit = set(expected_tools).issubset(called)
            intent_hit = result["intent"] == question.get("expect_intent")
            created = len(list_tickets()) > before or result.get("action_taken") is True
            expect_created = question.get("expect_ticket_created", False)
            safety_hit = created is expect_created
            pending_hit = None
            if "expect_pending" in question:
                pending_hit = (result.get("pending_action") is not None) is question["expect_pending"]
            escalation_hit = None
            if "expect_escalation" in question:
                recommended = bool(result.get("escalation") and result["escalation"].get("recommended"))
                escalation_hit = recommended is question["expect_escalation"]
            rows.append(
                {
                    "id": question["id"],
                    "kind": question["kind"],
                    "intent": result["intent"],
                    "intent_hit": intent_hit,
                    "keyword_recall": round(keyword_recall, 3),
                    "missing_keywords": [keyword for keyword in keywords if keyword not in hits],
                    "citation_hit": citation_hit,
                    "cited_documents": sorted(cited),
                    "tool_hit": tool_hit,
                    "tools": called,
                    "safety_hit": safety_hit,
                    "pending_hit": pending_hit,
                    "escalation_hit": escalation_hit,
                    "latency_seconds": round(elapsed, 4),
                }
            )
    finally:
        client.close()

    ablation = []
    for question in questions:
        if question["kind"] != "policy_qa" or not question.get("expected_documents"):
            continue
        for k in (2, 5):
            found = {hit["document_id"] for hit in search(question["message"], k=k)}
            expected = set(question["expected_documents"])
            ablation.append(
                {
                    "id": question["id"],
                    "k": k,
                    "hit": expected <= found,
                    "document_recall": round(len(expected & found) / len(expected), 3),
                    "found": sorted(found),
                }
            )

    def rate(items: list[dict], field: str) -> float:
        usable = [item for item in items if item.get(field) is not None]
        return mean([1.0 if item[field] else 0.0 for item in usable])

    latencies = [row["latency_seconds"] for row in rows]
    agentic = [row for row in rows if row["kind"] == "agentic"]
    clarify_rows = [row for row in rows if row["kind"] in {"clarify", "out_of_scope"}]
    summary = {
        "questions": len(rows),
        "groundedness_mean_keyword_recall": round(mean([row["keyword_recall"] for row in rows]), 3),
        "exact_keyword_match_rate": round(mean([1.0 if row["keyword_recall"] == 1 else 0.0 for row in rows]), 3),
        "citation_accuracy": round(rate(rows, "citation_hit"), 3),
        "tool_selection_accuracy": round(rate(rows, "tool_hit"), 3),
        "intent_accuracy": round(rate(rows, "intent_hit"), 3),
        "workflow_completion_rate": round(
            mean(
                [
                    1.0
                    if row["keyword_recall"] == 1 and row["tool_hit"] and row["intent_hit"] and row["citation_hit"] is not False
                    else 0.0
                    for row in agentic
                ]
            ),
            3,
        ),
        "clarification_accuracy": round(rate(clarify_rows, "intent_hit"), 3),
        "escalation_accuracy": round(rate(rows, "escalation_hit"), 3),
        "action_safety_pass_rate": round(rate(rows, "safety_hit"), 3),
        "latency_p50_seconds": round(percentile(latencies, 50), 4),
        "latency_p95_seconds": round(percentile(latencies, 95), 4),
        "ablation_k2_document_hit_rate": round(mean([1.0 if item["hit"] else 0.0 for item in ablation if item["k"] == 2]), 3),
        "ablation_k5_document_hit_rate": round(mean([1.0 if item["hit"] else 0.0 for item in ablation if item["k"] == 5]), 3),
        "ablation_k2_mean_document_recall": round(mean([item["document_recall"] for item in ablation if item["k"] == 2]), 3),
        "ablation_k5_mean_document_recall": round(mean([item["document_recall"] for item in ablation if item["k"] == 5]), 3),
    }
    return {"summary": summary, "rows": rows, "ablation": ablation}


def write_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evaluation results",
        "",
        "LLM synthesis was off. Tool choice and wording come from the orchestrator and the retrieved policy text.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Per question", "", "| ID | Intent | Keywords | Citations | Tools | Safety | Seconds |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in payload["rows"]:
        lines.append(
            f"| {row['id']} | {row['intent']} ({'ok' if row['intent_hit'] else 'miss'}) | {row['keyword_recall']} | "
            f"{row['citation_hit']} | {'ok' if row['tool_hit'] else 'miss'} | {'ok' if row['safety_hit'] else 'miss'} | {row['latency_seconds']} |"
        )
    misses = [row for row in payload["rows"] if row["missing_keywords"] or not row["tool_hit"] or row["citation_hit"] is False]
    if misses:
        lines.extend(["", "## Misses"])
        for row in misses:
            lines.append(
                f"- {row['id']}: missing keywords {row['missing_keywords'] or '[]'}; "
                f"cited {row['cited_documents']}; tools {row['tools']}"
            )
    lines.extend(
        [
            "",
            "## Ablation",
            "",
            "Same policy questions, comparing whether every expected document id appears in the top k chunks.",
            "",
            f"- k=2 full-hit rate: {summary['ablation_k2_document_hit_rate']}; mean document recall: {summary['ablation_k2_mean_document_recall']}",
            f"- k=5 full-hit rate: {summary['ablation_k5_document_hit_rate']}; mean document recall: {summary['ablation_k5_mean_document_recall']}",
            "",
            "Full hit means every expected document id appeared. Recall is the fraction of those ids in the top k.",
            "These numbers use one search of the raw question. The agent issues a separate search per topic, which is why its citation accuracy is higher.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    payload = evaluate()
    RESULTS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    RESULTS_MD.write_text(write_markdown(payload))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
