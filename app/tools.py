"""Tool implementations. The agent reaches these only through the MCP server."""

from __future__ import annotations

from datetime import datetime

from app.config import retrieval_k
from app.data_store import add_ticket, as_of, find_employee, public_employee, selected_benefit_details
from app.rag.store import get_section, search

INTERNATIONAL_MARKERS = (
    "international",
    "abroad",
    "overseas",
    "another country",
    "other country",
    "outside the united states",
    "outside the us",
    "canada",
    "mexico",
    "india",
    "united kingdom",
    "france",
    "germany",
    "spain",
    "japan",
    "australia",
    "brazil",
    "ireland",
)


def _employee_or_error(employee_id: str | None = None, name: str | None = None):
    if not employee_id and not name:
        return None, {"error": "missing_employee", "message": "Provide an employee id or name."}
    found = find_employee(employee_id=employee_id, name=name)
    if found is None:
        return None, {
            "error": "employee_not_found",
            "employee_id": employee_id,
            "name": name,
            "message": "No synthetic employee matches that id or name.",
        }
    if isinstance(found, list):
        return None, {
            "error": "ambiguous_employee",
            "matches": [{"employee_id": row["employee_id"], "full_name": row["full_name"]} for row in found],
        }
    return found, None


def _years_of_service(hire_date: str) -> float:
    hired = datetime.strptime(hire_date, "%Y-%m-%d").date()
    return (as_of() - hired).days / 365.25


def search_policy_documents(query: str, top_k: int | None = None, document_id: str | None = None) -> dict:
    k = int(top_k or retrieval_k())
    doc = str(document_id).strip() if document_id else None
    citations = search(query, k=k, document_id=doc or None)
    return {"query": query, "top_k": k, "document_id": doc, "citations": citations}


def get_policy_section(document_id: str, section: str) -> dict:
    return get_section(str(document_id), section or "")


def lookup_employee_profile(employee_id: str | None = None, name: str | None = None) -> dict:
    row, error = _employee_or_error(employee_id, name)
    if error:
        return error
    return {"employee": public_employee(row)}


def check_pto_balance(employee_id: str, requested_hours: float | None = None) -> dict:
    row, error = _employee_or_error(employee_id=str(employee_id))
    if error:
        return error
    balance = row["pto_balance"]
    available = float(balance["available_hours"])
    requested = None if requested_hours is None else float(requested_hours)
    eligible = row["employment_type"].lower() not in {"temporary", "contractor", "contract"}
    return {
        "employee_id": row["employee_id"],
        "full_name": row["full_name"],
        "employment_type": row["employment_type"],
        "pto_eligible": eligible,
        "manager_name": row["manager_name"],
        "available_hours": available,
        "used_ytd_hours": balance["used_ytd_hours"],
        "annual_accrual_hours": balance["annual_accrual_hours"],
        "requested_hours": requested,
        "sufficient_balance": None if requested is None else available + 1e-6 >= requested and eligible,
        "as_of": as_of().isoformat(),
    }


def lookup_benefits_status(employee_id: str) -> dict:
    row, error = _employee_or_error(employee_id=str(employee_id))
    if error:
        return error
    benefits = row["benefits_elections"]
    return {
        "employee_id": row["employee_id"],
        "full_name": row["full_name"],
        "employment_type": row["employment_type"],
        "hire_date": row["hire_date"],
        "years_of_service": round(_years_of_service(row["hire_date"]), 2),
        "benefits": benefits,
        **selected_benefit_details(row),
    }


def _location_kind(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in INTERNATIONAL_MARKERS):
        return "international"
    domestic = (
        "another state",
        "other state",
        "different state",
        "out of state",
        "u.s. state",
        "us state",
        "another u.s. state",
        "domestic",
        "different location",
        "another location",
        "within the united states",
        "within the u.s.",
        "within the us",
    )
    if any(marker in lowered for marker in domestic):
        return "domestic_other_state"
    return "unspecified"


def check_policy_compliance(
    employee_id: str,
    scenario: str,
    requested_hours: float | None = None,
    duration_weeks: float | None = None,
    duration_days: float | None = None,
    location_text: str | None = None,
    expense_item: str | None = None,
    benefits_topic: str | None = None,
) -> dict:
    row, error = _employee_or_error(employee_id=str(employee_id))
    if error:
        return error
    scenario = (scenario or "").lower().strip()
    if scenario == "pto":
        return _pto_decision(row, requested_hours)
    if scenario in {"remote_location", "remote_eligibility", "remote"}:
        return _remote_decision(row, scenario, duration_weeks, duration_days, location_text or "")
    if scenario == "expense":
        return _expense_decision(row, expense_item or "")
    if scenario == "benefits":
        return _benefits_decision(row, benefits_topic or "general")
    return {"error": "unknown_scenario", "scenario": scenario}


def _pto_decision(row: dict, requested_hours: float | None) -> dict:
    balance = check_pto_balance(row["employee_id"], requested_hours)
    days = None if requested_hours is None else float(requested_hours) / 8.0
    if not balance["pto_eligible"]:
        decision = "not_eligible"
        notice = None
    elif requested_hours is None:
        decision = "needs_duration"
        notice = None
    elif not balance["sufficient_balance"]:
        decision = "insufficient_balance"
        notice = 5 if days is not None and days <= 3 else 10
    else:
        decision = "eligible_to_request"
        notice = 5 if days is not None and days <= 3 else 10
    reasons = []
    if decision == "not_eligible":
        reasons.append("Policy 200 covers full-time and part-time regular employees. Temporary employees and contractors are not eligible for PTO unless a contract says otherwise.")
    else:
        reasons.append("Policy 200 requires manager approval for every PTO request. The employee submits the request in WorkDay.")
        if days is not None and days <= 3:
            reasons.append("A request of 1-3 days needs at least 5 business days' notice.")
        elif days is not None:
            reasons.append("A request of 4 or more days needs at least 10 business days' notice.")
        if decision == "insufficient_balance":
            reasons.append("The requested hours exceed the available PTO balance on the employee record.")
        elif decision == "eligible_to_request":
            reasons.append("The requested hours are within the available PTO balance on the employee record. This is not an approval.")
    return {
        "scenario": "pto",
        "decision": decision,
        "policy_refs": ["200"],
        "notice_business_days_required": notice,
        "reasons": reasons,
        **balance,
    }


def _remote_decision(row: dict, scenario: str, weeks: float | None, days: float | None, location_text: str) -> dict:
    kind = _location_kind(location_text)
    if weeks is None and days is not None:
        weeks = float(days) / 7.0
    reasons = []
    escalate = []
    if kind == "international":
        decision = "escalate_international"
        escalate = ["People & Culture", "Legal"]
        reasons.append("Policy 300 section 5.2 says work outside the United States is generally not permitted. Any request must be escalated to People & Culture and Legal. Approval is rare.")
        if not row["international_remote_approved"]:
            reasons.append("The employee record has international remote work marked as not approved.")
        if not row["remote_work_eligibility"]:
            reasons.append("The employee record also marks this role as not remote-eligible.")
    elif not row["remote_work_eligibility"]:
        decision = "not_remote_eligible"
        reasons.append("The employee record marks this role as not remote-eligible. Policy 300 limits remote work to roles designated remote-eligible, with manager and department-head approval.")
    elif (weeks or 0) > 4:
        decision = "requires_change_of_work_location"
        escalate = ["People & Culture", "Finance"]
        reasons.append("Policy 300 section 5.1 limits temporary domestic remote work to 4 weeks per calendar year.")
        reasons.append("Policy 300 section 5.3 requires a Change of Work Location request in WorkDay when work from another U.S. state exceeds 4 weeks, followed by manager approval and a People & Culture and Finance tax review.")
        used = row["temporary_remote_days_used_ytd"]
        limit = row["temporary_remote_days_limit"]
        reasons.append(
            f"The employee record shows {used} of {limit} temporary remote days used. "
            f"The requested duration exceeds that counter as well as the 4-week policy cap."
        )
        if not row["change_of_work_location_filed"]:
            reasons.append("No Change of Work Location is on file for this employee.")
    elif kind == "domestic_other_state" or (weeks is not None and weeks <= 4):
        decision = "manager_approval_for_temporary_remote"
        reasons.append("Policy 300 section 5.1 allows a temporary domestic location for no more than 4 weeks per calendar year, with manager approval and tax or legal review.")
        remaining = row["temporary_remote_days_limit"] - row["temporary_remote_days_used_ytd"]
        reasons.append(f"The employee record has {remaining} temporary remote days remaining out of {row['temporary_remote_days_limit']}.")
    else:
        decision = "already_classified"
        reasons.append(
            f"The employee record lists remote status as {row['remote_work_status']} and remote eligibility as {row['remote_work_eligibility']}."
        )
        reasons.append("Policy 300 allows hybrid employees up to 3 remote days per week. Fully remote status is a separate approval. A different city, state, or country needs the location rules in section 5.")
    return {
        "scenario": "remote_location" if scenario != "remote_eligibility" else "remote_eligibility",
        "decision": decision,
        "policy_refs": ["300", "600"],
        "location_kind": kind,
        "duration_weeks": weeks,
        "remote_work_status": row["remote_work_status"],
        "remote_work_eligibility": row["remote_work_eligibility"],
        "work_location": row["work_location"],
        "state_of_residence": row["state_of_residence"],
        "international_remote_approved": row["international_remote_approved"],
        "change_of_work_location_filed": row["change_of_work_location_filed"],
        "temporary_remote_days_used_ytd": row["temporary_remote_days_used_ytd"],
        "temporary_remote_days_limit": row["temporary_remote_days_limit"],
        "manager_name": row["manager_name"],
        "escalate_to": escalate,
        "reasons": reasons,
        "employee_id": row["employee_id"],
        "full_name": row["full_name"],
    }


def _expense_decision(row: dict, item: str) -> dict:
    lowered = item.lower()
    stipend = row["home_office_stipend"]
    remoteish = row["remote_work_status"].lower() in {"hybrid", "fully remote"} and row["remote_work_eligibility"]
    remaining = max(0, 500 - float(stipend.get("amount_used") or 0))
    if any(word in lowered for word in ("desk", "internet", "coffee", "clothing")):
        decision = "not_reimbursable"
        reasons = ["Policy 400 lists desks, internet service, coffee makers, and general office supplies as ineligible for the home office stipend."]
    elif "laptop" in lowered:
        decision = "company_issued"
        reasons = [
            "Policy 800 issues a company laptop to full-time employees. A personal laptop is not an eligible home-office stipend item in Policy 400.",
            "Eligible stipend items are a monitor, keyboard, mouse, webcam, headset, desk lamp, and ergonomic chair.",
        ]
    elif any(word in lowered for word in ("chair", "monitor", "keyboard", "mouse", "webcam", "headset", "lamp")):
        if not remoteish:
            decision = "not_eligible_for_stipend"
            reasons = ["Policy 400 limits the one-time $500 home office stipend to approved remote or hybrid employees."]
        elif stipend.get("used") and remaining <= 0:
            decision = "stipend_exhausted"
            reasons = ["The employee record shows the one-time home office stipend has already been used. Policy 400 allows the stipend once."]
        else:
            decision = "reimbursable_from_stipend"
            reasons = [
                "Policy 400 includes an ergonomic chair and listed peripherals in the one-time $500 home office stipend for remote or hybrid employees.",
                "The employee buys the item, then submits one Concur report under Home Office Stipend with itemized receipts.",
            ]
    elif any(word in lowered for word in ("flight", "hotel", "lodging", "meal", "travel")):
        decision = "travel_preapproval_required"
        reasons = [
            "Policy 900 requires manager pre-approval and booking through CorporateTravel at least 14 days ahead.",
            "Economy class is required. Policy 400 states economy for flights under 6 hours. Policy 900 states economy for all flights, with premium economy or business class only if a VP approves a flight over 8 hours.",
            "Hotel rates should not exceed $300 per night before taxes. Meals use a $75 daily per diem inside the United States.",
        ]
    else:
        decision = "needs_item_review"
        reasons = [
            "Policy 400 requires manager pre-approval, an itemized receipt for expenses over $25, and a Concur report within 30 days.",
            "Expenses submitted more than 60 days after they were incurred may be denied.",
        ]
    return {
        "scenario": "expense",
        "decision": decision,
        "policy_refs": ["400", "800", "900"],
        "item": item,
        "employment_type": row["employment_type"],
        "remote_work_status": row["remote_work_status"],
        "stipend_used": bool(stipend.get("used")),
        "stipend_amount_used": stipend.get("amount_used"),
        "stipend_remaining": remaining,
        "manager_name": row["manager_name"],
        "reasons": reasons,
        "employee_id": row["employee_id"],
        "full_name": row["full_name"],
    }


def _benefits_decision(row: dict, topic: str) -> dict:
    benefits = row["benefits_elections"]
    topic = topic.lower()
    years = _years_of_service(row["hire_date"])
    reasons = []
    if row["employment_type"].lower() in {"temporary", "contractor", "contract"} or not benefits.get("benefits_eligible"):
        decision = "not_eligible"
        reasons.append("Policy 500 says temporary and contract employees are not eligible for company-sponsored benefits.")
    elif "hsa" in topic:
        plan = (benefits.get("medical_plan") or "") 
        if "HDHP" in plan.upper() or "high-deductible" in plan.lower():
            decision = "hsa_eligible"
            reasons.append("Policy 500 offers an HSA to employees on a high-deductible health plan and contributes $500 annually.")
            reasons.append(
                "The employee record shows HSA enrollment as "
                + ("yes." if benefits.get("hsa_enrolled") else "not enrolled. Enrollment is separate from eligibility.")
            )
        else:
            decision = "hsa_not_eligible"
            reasons.append("Policy 500 limits the HSA to employees enrolled in a high-deductible health plan. This record uses a different medical plan. An FSA may still be available.")
    elif "tuition" in topic:
        if row["employment_type"].lower() != "full-time":
            decision = "tuition_not_eligible"
            reasons.append("Policy 500 limits tuition reimbursement to full-time employees after one year of employment.")
        elif years < 1:
            decision = "tuition_not_yet"
            reasons.append("Policy 500 requires one year of full-time employment before tuition reimbursement of up to $5,000 per calendar year.")
        else:
            decision = "tuition_eligible"
            reasons.append("Policy 500 allows up to $5,000 per calendar year for job-related courses or degrees after one year of full-time employment, with manager approval.")
    elif "401" in topic:
        decision = "retirement_eligible" if benefits.get("benefits_eligible") else "not_eligible"
        reasons.append("Policy 500 matches 100% of 401k contributions up to 4% of eligible compensation after 30 days of employment. The match vests immediately.")
        reasons.append(f"The employee record shows a contribution rate of {benefits.get('401k_contribution_percent')}%.")
    else:
        decision = "enrolled" if benefits.get("benefits_enrolled") else "eligible_not_enrolled"
        if row["employment_type"].lower() == "full-time":
            reasons.append("Policy 500 makes full-time employees eligible for benefits on the first of the month after 30 days of employment.")
        elif row["employment_type"].lower() == "part-time":
            reasons.append("Policy 500 makes part-time employees scheduled for at least 20 hours eligible for medical, dental, and vision on the first of the month after 60 days.")
        reasons.append(
            f"The employee record shows medical plan {benefits.get('medical_plan') or 'none'}, "
            f"dental {benefits.get('dental')}, vision {benefits.get('vision')}, "
            f"and 401k contribution {benefits.get('401k_contribution_percent')}%."
        )
    return {
        "scenario": "benefits",
        "decision": decision,
        "policy_refs": ["500"],
        "topic": topic,
        "years_of_service": round(years, 2),
        "reasons": reasons,
        "employee_id": row["employee_id"],
        "full_name": row["full_name"],
        "employment_type": row["employment_type"],
        "benefits": benefits,
    }


def draft_hr_email(employee_id: str, recipient: str, purpose: str, details: str) -> dict:
    row, error = _employee_or_error(employee_id=str(employee_id))
    if error:
        return error
    recipient = recipient or "manager"
    if recipient == "hr":
        to_line = "People & Culture <hr@innovatech.example.com>"
    elif recipient == "ethics":
        to_line = "Ethics Hotline <ethics@innovatech.example.com>"
    else:
        to_line = f"{row['manager_name']} (manager)"
    subject = {
        "pto": f"PTO request from {row['full_name']}",
        "remote": f"Remote location request from {row['full_name']}",
        "expense": f"Expense question from {row['full_name']}",
        "benefits": f"Benefits question from {row['full_name']}",
        "conduct": f"Workplace concern from {row['full_name']}",
    }.get(purpose, f"HR question from {row['full_name']}")
    body = (
        f"Hello,\n\n"
        f"I am {row['full_name']} (employee {row['employee_id']}), {row['role']} in {row['department']}.\n\n"
        f"{details.strip()}\n\n"
        f"This note is a draft. It has not been sent.\n\n"
        f"Thank you,\n{row['full_name']}\n{row['email']}"
    )
    return {
        "status": "draft_only",
        "sent": False,
        "to": to_line,
        "subject": subject,
        "body": body,
        "employee_id": row["employee_id"],
    }


def create_mock_hr_ticket(
    employee_id: str,
    category: str,
    summary: str,
    confirm: bool = False,
) -> dict:
    row, error = _employee_or_error(employee_id=str(employee_id))
    if error:
        return error
    draft = {
        "employee_id": row["employee_id"],
        "employee_name": row["full_name"],
        "category": category or "general",
        "summary": (summary or "").strip()[:1000],
    }
    if confirm is not True:
        return {
            "status": "confirmation_required",
            "stored": False,
            "message": "No ticket was stored. Ask the employee to confirm before calling this tool with confirm=true.",
            "draft": draft,
        }
    stored = add_ticket(draft)
    return {"status": "created", "stored": True, "ticket": stored}


TOOL_SPECS = [
    {
        "name": "search_policy_documents",
        "description": "Retrieve top-k Innovatech policy chunks with document id, section, source, and snippet.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language policy question."},
                "top_k": {"type": "integer", "description": "Number of chunks to return. Defaults to 4."},
                "document_id": {"type": "string", "description": "Optional policy id filter such as 200 or 300."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_policy_section",
        "description": "Return a named section from one policy document.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
                "section": {"type": "string", "description": "Section title or phrase to find inside the document."},
            },
            "required": ["document_id", "section"],
        },
    },
    {
        "name": "lookup_employee_profile",
        "description": "Look up a synthetic employee profile, manager, location, and equipment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "name": {"type": "string", "description": "Full name or a message that contains the employee name."},
            },
        },
    },
    {
        "name": "check_pto_balance",
        "description": "Read a synthetic PTO balance and compare it with requested hours.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "requested_hours": {"type": "number"},
            },
            "required": ["employee_id"],
        },
    },
    {
        "name": "lookup_benefits_status",
        "description": "Read synthetic benefits elections and employment type.",
        "inputSchema": {
            "type": "object",
            "properties": {"employee_id": {"type": "string"}},
            "required": ["employee_id"],
        },
    },
    {
        "name": "check_policy_compliance",
        "description": "Apply written policy thresholds to one synthetic employee. Does not approve or change records.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "scenario": {"type": "string", "description": "pto, remote_location, remote_eligibility, expense, or benefits."},
                "requested_hours": {"type": "number"},
                "duration_weeks": {"type": "number"},
                "duration_days": {"type": "number"},
                "location_text": {"type": "string"},
                "expense_item": {"type": "string"},
                "benefits_topic": {"type": "string"},
            },
            "required": ["employee_id", "scenario"],
        },
    },
    {
        "name": "draft_hr_email",
        "description": "Draft an email. This never sends mail.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "recipient": {"type": "string", "description": "manager, hr, or ethics."},
                "purpose": {"type": "string"},
                "details": {"type": "string"},
            },
            "required": ["employee_id", "recipient", "purpose", "details"],
        },
    },
    {
        "name": "create_mock_hr_ticket",
        "description": "Store a mock HR ticket only when confirm is true. Otherwise return a draft and store nothing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "category": {"type": "string"},
                "summary": {"type": "string"},
                "confirm": {"type": "boolean"},
            },
            "required": ["employee_id", "category", "summary"],
        },
    },
]

_DISPATCH = {
    "search_policy_documents": search_policy_documents,
    "get_policy_section": get_policy_section,
    "lookup_employee_profile": lookup_employee_profile,
    "check_pto_balance": check_pto_balance,
    "lookup_benefits_status": lookup_benefits_status,
    "check_policy_compliance": check_policy_compliance,
    "draft_hr_email": draft_hr_email,
    "create_mock_hr_ticket": create_mock_hr_ticket,
}


def call_tool(name: str, arguments: dict | None) -> dict:
    function = _DISPATCH.get(name)
    if function is None:
        return {"error": "unknown_tool", "tool": name}
    arguments = arguments or {}
    try:
        return function(**arguments)
    except TypeError as exc:
        return {"error": "invalid_arguments", "tool": name, "detail": str(exc)}
