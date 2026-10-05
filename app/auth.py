"""Demo sign-in. Passwords live in mock_data/passwords.json, one per employee."""

from __future__ import annotations

import hmac
import json
import re

from app.config import passwords_path
from app.data_store import employees, find_employee

_EMPLOYEE_ID = re.compile(r"\bemployee\s*#?\s*(\d{3})\b", re.I)


def load_passwords() -> dict[str, str]:
    payload = json.loads(passwords_path().read_text())
    return {str(row["employee_id"]): str(row["password"]) for row in payload["employees"]}


def verify_password(employee_id: str, password: str) -> bool:
    stored = load_passwords().get(str(employee_id).strip())
    if not stored or not password:
        return False
    return hmac.compare_digest(stored, password)


def other_employee_requested(message: str, actor_id: str) -> bool:
    """True when the message asks about a different employee's record."""
    actor_id = str(actor_id)
    actor = find_employee(employee_id=actor_id)
    if not isinstance(actor, dict):
        return True
    for match in _EMPLOYEE_ID.finditer(message or ""):
        if match.group(1) != actor_id:
            return True
    actor_first = actor["full_name"].split()[0].lower()
    lowered = message or ""
    for row in employees():
        if str(row["employee_id"]) == actor_id:
            continue
        if row["full_name"].lower() in lowered.lower():
            return True
        first = row["full_name"].split()[0].lower()
        if first != actor_first and re.search(rf"\b{re.escape(first)}\b", lowered, re.I):
            return True
    return False
