"""Optional final-answer rewrite. Tool selection stays deterministic either way."""

from __future__ import annotations

import os

import httpx

SYSTEM = (
    "You rewrite HR answers for Innovatech Solutions. Use only the evidence in the draft and tool results. "
    "Do not add policy rules, numbers, or approvals that are not in the evidence. Keep citations like [200 | section]. "
    "Label recommendations separately from policy facts. If the draft says an action was not taken, do not claim it was. "
    "If evidence is missing, say so."
)


def _api_key() -> str:
    if os.environ.get("LLM_DISABLED") == "1":
        return ""
    return (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or ""
    )


def _base_url() -> str:
    if os.environ.get("LLM_BASE_URL"):
        return os.environ["LLM_BASE_URL"].rstrip("/")
    if os.environ.get("GROQ_API_KEY") and not os.environ.get("LLM_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        return "https://api.groq.com/openai/v1"
    if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("LLM_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        return "https://openrouter.ai/api/v1"
    return "https://api.openai.com/v1"


def _model() -> str:
    return os.environ.get("LLM_MODEL") or "gpt-4o-mini"


def synthesis_enabled() -> bool:
    return bool(_api_key())


def rewrite(question: str, draft: str) -> str | None:
    key = _api_key()
    if not key:
        return None
    try:
        response = httpx.post(
            f"{_base_url()}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": _model(),
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"Employee question:\n{question}\n\nEvidence draft:\n{draft}",
                    },
                ],
            },
            timeout=40,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None
