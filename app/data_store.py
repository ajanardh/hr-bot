"""Synthetic employee, location-request, and mock-ticket storage."""

from __future__ import annotations

import json
import re
import threading
from datetime import date, datetime

from app.config import employees_path, tickets_path

_LOCK = threading.Lock()
_EMPLOYEES: dict | None = None


def load_database() -> dict:
    global _EMPLOYEES
    if _EMPLOYEES is None:
        _EMPLOYEES = json.loads(employees_path().read_text())
    return _EMPLOYEES


def employees() -> list[dict]:
    return load_database()["employees"]


def location_requests() -> list[dict]:
    return load_database().get("change_of_work_location_requests", [])


def healthcare_plans() -> dict:
    return load_database().get("healthcare_plans", {})


def retirement_plan() -> dict:
    return load_database().get("retirement_plan", {})


def _money(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = int(value) if float(value).is_integer() else value
        return f"${number:,}" if isinstance(number, int) else f"${number}"
    return str(value)


def plan_fact(message: str, benefits: dict | None = None) -> str:
    """A direct fact from the healthcare or 401k plan data when the question asks for one."""
    text = (message or "").lower().replace("401(k)", "401k").replace("401 (k)", "401k")
    plans = healthcare_plans().get("medical_plans") or []
    selected = None
    for plan in plans:
        name = str(plan.get("plan_name") or "").lower()
        if name and name in text:
            selected = plan
            break
    if selected is None and ("hdhp" in text or "high deductible" in text):
        selected = next((plan for plan in plans if plan.get("plan_id") == "HDHP"), None)
    if selected is None and re.search(r"\bhmo\b", text):
        selected = next((plan for plan in plans if plan.get("plan_id") == "HMO_STANDARD"), None)
    asks_cost = re.search(r"\b(deductible\w*|copay\w*|premium\w*|how much|cost)\b", text)
    if selected is None and benefits and asks_cost and re.search(r"\b(my|i|me)\b", text):
        plan_id = benefits.get("medical_plan_id")
        selected = next((plan for plan in plans if plan.get("plan_id") == plan_id), None)
    sentences: list[str] = []
    if selected and (asks_cost or re.search(r"\b(ppo|hmo|hdhp)\b", text)):
        visit = selected.get("primary_care_copay")
        visit_text = _money(visit) if isinstance(visit, (int, float)) else str(visit).lower()
        sentences.append(
            f"{selected.get('plan_name')} has an individual deductible of {_money(selected.get('deductible_individual'))} "
            f"and a family deductible of {_money(selected.get('deductible_family'))}. "
            f"A primary care visit is {visit_text}."
        )
        if re.search(r"\bpremium", text):
            monthly = selected.get("premiums_monthly") or {}
            sentences.append(
                "Monthly premiums are "
                f"{_money(monthly.get('employee_only'))} for you, "
                f"{_money(monthly.get('employee_spouse'))} with a spouse, "
                f"{_money(monthly.get('employee_children'))} with children, "
                f"and {_money(monthly.get('family'))} for a family."
            )
    parameters = retirement_plan().get("plan_parameters") or {}
    if re.search(r"\b(401k|vest\w*|company match|matching)\b", text) and parameters.get("match_formula"):
        sentences.append(
            f"The company match is {parameters.get('match_formula')}. Vesting is {parameters.get('match_vesting')}."
        )
    return " ".join(sentences)


def selected_benefit_details(row: dict) -> dict:
    """Plan facts for the healthcare and 401k choices stored on this employee."""
    benefits = row.get("benefits_elections") or {}
    employee_id = str(row.get("employee_id"))
    medical = None
    for plan in healthcare_plans().get("medical_plans") or []:
        if plan.get("plan_id") == benefits.get("medical_plan_id"):
            medical = {
                "plan_id": plan.get("plan_id"),
                "plan_name": plan.get("plan_name"),
                "carrier": plan.get("carrier"),
                "hsa_eligible": plan.get("hsa_eligible"),
                "deductible_individual": plan.get("deductible_individual"),
                "primary_care_copay": plan.get("primary_care_copay"),
            }
            break
    funds = {
        item.get("fund_id"): item.get("fund_name")
        for item in retirement_plan().get("investment_options") or []
    }
    allocation = [
        {
            "fund_id": piece.get("fund_id"),
            "fund_name": funds.get(piece.get("fund_id"), piece.get("fund_id")),
            "percent": piece.get("percent"),
        }
        for piece in benefits.get("401k_investment_allocation") or []
    ]
    retirement = retirement_plan()

    def owned(records: list | None) -> list:
        return [item for item in records or [] if str(item.get("employee_id")) == employee_id]

    return {
        "medical_plan_details": medical,
        "dental_plan": healthcare_plans().get("dental_plan") if benefits.get("dental") else None,
        "vision_plan": healthcare_plans().get("vision_plan") if benefits.get("vision") else None,
        "401k_investment_allocation": allocation,
        "401k_loans": owned(retirement.get("loan_records")),
        "401k_hardship_withdrawals": owned(retirement.get("hardship_withdrawal_records")),
        "401k_rollovers": owned(retirement.get("rollover_records")),
    }


def as_of() -> date:
    raw = load_database().get("metadata", {}).get("as_of_date", "2023-10-26")
    return datetime.strptime(raw, "%Y-%m-%d").date()


def find_employee(employee_id: str | None = None, name: str | None = None) -> dict | list | None:
    rows = employees()
    if employee_id:
        target = str(employee_id).strip()
        for row in rows:
            if str(row["employee_id"]) == target:
                return row
        return None
    if not name:
        return None
    query = name.lower()
    full = [row for row in rows if row["full_name"].lower() in query]
    if len(full) == 1:
        return full[0]
    if len(full) > 1:
        return full
    first = []
    for row in rows:
        given = row["full_name"].split()[0].lower()
        if f" {given} " in f" {query} ":
            first.append(row)
    if len(first) == 1:
        return first[0]
    if len(first) > 1:
        return first
    return None


def public_employee(row: dict) -> dict:
    requests = [
        item for item in location_requests() if str(item.get("employee_id")) == str(row["employee_id"])
    ]
    return {
        "employee_id": row["employee_id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "role": row["role"],
        "department": row["department"],
        "manager_id": row["manager_id"],
        "manager_name": row["manager_name"],
        "employment_type": row["employment_type"],
        "hire_date": row["hire_date"],
        "office_location": row["office_location"],
        "work_location": row["work_location"],
        "state_of_residence": row["state_of_residence"],
        "remote_work_status": row["remote_work_status"],
        "remote_work_eligibility": row["remote_work_eligibility"],
        "remote_work_days_used_ytd": row["remote_work_days_used_ytd"],
        "temporary_remote_days_used_ytd": row["temporary_remote_days_used_ytd"],
        "temporary_remote_days_limit": row["temporary_remote_days_limit"],
        "international_remote_approved": row["international_remote_approved"],
        "change_of_work_location_filed": row["change_of_work_location_filed"],
        "pto_balance": row["pto_balance"],
        "benefits_elections": row["benefits_elections"],
        "equipment_issued": row["equipment_issued"],
        "home_office_stipend": row["home_office_stipend"],
        "location_requests": requests,
    }


def list_tickets() -> list[dict]:
    path = tickets_path()
    if not path.exists():
        return []
    raw = path.read_text().strip() or "[]"
    return json.loads(raw)


def add_ticket(ticket: dict) -> dict:
    with _LOCK:
        path = tickets_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        current = []
        if path.exists() and path.read_text().strip():
            current = json.loads(path.read_text())
        stored = {
            "ticket_id": f"TICK-{len(current) + 1:04d}",
            "created_at": as_of().isoformat(),
            "status": "open_mock",
            **ticket,
        }
        current.append(stored)
        path.write_text(json.dumps(current, indent=2) + "\n")
        return stored
