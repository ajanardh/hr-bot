"""Orchestrator: classify the request, call MCP tools, and answer from tool evidence."""

from __future__ import annotations

import re

from app.auth import other_employee_requested
from app.config import retrieval_k
from app.data_store import plan_fact
from app.llm import rewrite
from app.mcp_client import McpClient, McpError
from app.places import mentions_international
from app.rag.store import STOP, missing_terms, tokenize

NUMBERS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

OUT_OF_SCOPE = re.compile(
    r"\b(weather|stock price|recipe|super bowl|capital of|write (me )?(python|javascript|code)|"
    r"who won|tell me a joke|translate this)\b",
    re.I,
)
SENSITIVE = re.compile(r"harass|discriminat|bully|retaliat|workplace violence|ethic|hostile work", re.I)
TICKET = re.compile(r"\b(create|file|open|submit)\b.{0,40}\b(ticket|hr case|case)\b", re.I)
PERSONAL = re.compile(
    r"\b(can i|could i|am i|do i|my |i want|i need|i'd like|i would|for me|employee\s+\d+|can employee)\b",
    re.I,
)
EMPLOYEE_ID = re.compile(r"\bemployee\s*#?\s*(\d{3})\b", re.I)

CATEGORIES = [
    ("pto", re.compile(r"\b(pto|vacation|time off|sick|parental|bereavement|jury|holidays?|leave)\b", re.I)),
    ("remote", re.compile(r"\b(remote\w*|hybrid|another (?:u\.?s\.? )?state|other state|abroad|international\w*|work from)\b", re.I)),
    ("travel", re.compile(r"\b(travel|flight|hotel|lodging|per diem|airfare|mileage)\b", re.I)),
    ("expense", re.compile(r"\b(expense|reimburse\w*|stipend|concur|receipt|chair|desk)\b", re.I)),
    (
        "benefits",
        re.compile(
            r"\b(benefit\w*|401k|insurance|medical|dental|vision|hsa|fsa|tuition|wellness|eap|"
            r"healthcare|health|retirement|deductible\w*|copay\w*|coinsurance|premium\w*|ppo|hmo|hdhp|"
            r"cobra|enroll\w*|vest\w*|rollover|hardship|contribution\w*|prescription|pharmacy|"
            r"fidelity|wellnesscorp|dependent\w*|company match|matching)\b",
            re.I,
        ),
    ),
    ("security", re.compile(r"\b(password|mfa|vpn|phishing|encrypt\w*|confidential|security|incident|globalprotect)\b", re.I)),
    ("onboarding", re.compile(r"\b(onboard\w*|offboard\w*|resign\w*|new hire|day 1)\b", re.I)),
    ("equipment", re.compile(r"\b(equipment|laptop|monitor|badge)\b", re.I)),
    ("conduct", re.compile(r"\b(code of conduct|harassment|discrimination|ethics)\b", re.I)),
]

CATEGORY_QUERY = {
    "pto": "PTO accrual holidays sick leave parental bereavement notice approval WorkDay",
    "remote": "remote hybrid temporary another state international change of work location",
    "travel": "travel hotel flight per diem CorporateTravel lodging meals",
    "expense": "expense reimbursement home office stipend Concur receipt chair desk",
    "benefits": "benefits eligibility medical 401k HSA FSA tuition wellness",
    "security": "VPN GlobalProtect password MFA encryption confidential device",
    "onboarding": "onboarding offboarding day 1 resignation notice equipment return",
    "equipment": "laptop equipment return stipend home office IT",
    "conduct": "harassment discrimination reporting ethics retaliation",
}


class Agent:
    def __init__(self, mcp: McpClient) -> None:
        self.mcp = mcp

    def run(
        self,
        message: str,
        employee_id: str | None = None,
        confirm_action: bool = False,
        pending_action: dict | None = None,
        actor_employee_id: str | None = None,
        selected_step: dict | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        message = (message or "").strip()
        if actor_employee_id:
            actor_employee_id = str(actor_employee_id)
            employee_id = actor_employee_id
            if pending_action:
                arguments = dict(pending_action.get("arguments") or {})
                requested = arguments.get("employee_id")
                if requested and str(requested) != actor_employee_id:
                    return self._private_refusal()
                arguments["employee_id"] = actor_employee_id
                pending_action = {**pending_action, "arguments": arguments}
            step_employee = (selected_step or {}).get("employee_id")
            if step_employee and str(step_employee) != actor_employee_id:
                return self._private_refusal()
            if not selected_step and other_employee_requested(message, actor_employee_id):
                return self._private_refusal()
        if selected_step:
            return self._run_step(message, employee_id, selected_step)
        if not message and not (confirm_action and pending_action):
            return self._finish(message, "clarify", "Ask a question about Innovatech policy or an HR workflow.", [], [], None, None)
        if confirm_action and pending_action:
            return self._confirm(message, pending_action)
        if pending_action and re.search(r"\b(cancel|don't|do not|stop|no)\b", message, re.I):
            return self._finish(
                message,
                "ticket",
                "No ticket was stored. The draft was discarded.",
                [],
                [{"step": 1, "tool": None, "status": "cancelled", "output": {"stored": False}}],
                None,
                None,
            )

        action = _conversation_action(message)
        if action and _prior_topic(history):
            return self._continue_conversation(message, employee_id, history, action)

        message = _with_history(message, history)
        if actor_employee_id and other_employee_requested(message, actor_employee_id):
            return self._private_refusal()

        employee_id = employee_id or self._explicit_id(message)
        intent = self._classify(message)
        trace: list[dict] = []

        if intent == "out_of_scope":
            return self._finish(
                message,
                intent,
                "That request is outside the Innovatech policy corpus. I can help with PTO, holidays, remote work, "
                "expenses, benefits, security, equipment, onboarding, and workplace conduct. "
                "For anything else, contact People & Culture at hr@innovatech.example.com.",
                trace,
                [],
                None,
                None,
            )

        if _wants_location_rules(message):
            return self._location_rules(message, employee_id, trace)

        if intent == "clarify_pto":
            citations = self._search(trace, message)
            answer = "Tell me how many days you need. Nothing has been requested."
            left = self._pto_left_for(employee_id, trace)
            if left:
                answer = f"{left} {answer}"
            return self._finish(message, "clarify", answer, trace, citations, None, None)

        if intent == "clarify_remote":
            citations = self._search(trace, "remote hybrid eligibility another location")
            answer = (
                "Tell me where you want to work and how long you want to be there. "
                "I can then check the location rules. Nothing has been requested."
            )
            status = self._authorization_for(employee_id, trace)
            if status:
                answer = f"{status} {answer}"
            return self._finish(message, "clarify", answer, trace, citations, None, None)

        if intent == "ticket":
            return self._ticket(message, employee_id, trace)

        if intent == "conduct":
            return self._conduct(message, employee_id, trace)

        if intent in {"pto", "remote", "expense", "benefits"}:
            if not employee_id:
                employee_id = self._id_from_lookup(trace, message)
            if not employee_id:
                citations = self._search(trace, message)
                answer = (
                    "I can answer the general rule, and I still need you to be signed in before I use a personal record. "
                    + self._direct_policy(citations, message)
                )
                return self._finish(message, "clarify", answer, trace, citations, None, None)
            handler = {
                "pto": self._pto,
                "remote": self._remote,
                "expense": self._expense,
                "benefits": self._benefits,
            }[intent]
            return handler(message, employee_id, trace)

        return self._policy_qa(message, trace, intent, employee_id)

    def _private_refusal(self) -> dict:
        return self._finish(
            "",
            "private",
            "You can only look up your own employee record. Ask about your PTO, benefits, or a general policy, "
            "or sign in as the other person. I did not open anyone else's record.",
            [],
            [],
            None,
            None,
        )

    def _confirm(self, message: str, pending_action: dict) -> dict:
        trace: list[dict] = []
        if pending_action.get("tool") != "create_mock_hr_ticket":
            return self._finish(
                message,
                "ticket",
                "I did not take that action. Only a mock HR ticket can be confirmed, and nothing was stored.",
                trace,
                [],
                None,
                None,
            )
        arguments = dict(pending_action.get("arguments") or {})
        arguments["confirm"] = True
        output = self._call(trace, "create_mock_hr_ticket", arguments)
        if output.get("stored"):
            ticket = output.get("ticket", {})
            answer = (
                f"Mock ticket {ticket.get('ticket_id')} is stored with status {ticket.get('status')}. "
                "This does not approve the underlying request. People & Culture still has to review it."
            )
        else:
            answer = "The ticket was not stored. " + str(output.get("message") or output.get("error") or "")
        return self._finish(message, "ticket", answer, trace, [], None, None)

    def _ticket(self, message: str, employee_id: str | None, trace: list[dict]) -> dict:
        if not employee_id:
            employee_id = self._id_from_lookup(trace, message)
        if not employee_id:
            return self._finish(
                message,
                "clarify",
                "I can prepare a mock ticket after you tell me the employee id or name. Nothing has been stored.",
                trace,
                [],
                None,
                None,
            )
        profile = self._call(trace, "lookup_employee_profile", {"employee_id": employee_id})
        if profile.get("error"):
            return self._finish(message, "clarify", profile.get("message", "Employee not found."), trace, [], None, None)
        pending = {
            "tool": "create_mock_hr_ticket",
            "arguments": {
                "employee_id": employee_id,
                "category": "general",
                "summary": message[:800],
                "confirm": False,
            },
        }
        name = profile["employee"]["full_name"]
        answer = (
            f"I can open a mock HR ticket for {name} ({employee_id}). "
            "Nothing is stored until you confirm.\n\n"
            f"Draft summary: {message}\n\n"
            "Recommendation: confirm only if you want this mock case on file. Confirmation does not approve PTO, expenses, or remote work."
        )
        return self._finish(message, "ticket", answer, trace, [], None, pending)

    def _conduct(self, message: str, employee_id: str | None, trace: list[dict]) -> dict:
        citations = []
        citations += self._search(trace, "harassment discrimination reporting ethics hotline retaliation")
        citations += self._search(trace, "HR case handling escalation confirmation ethics", document_id="010")
        escalation = {
            "recommended": True,
            "team": "People & Culture and the Ethics Hotline",
            "reason": "Conduct concerns are reported to a manager, People & Culture, or ethics@innovatech.example.com. The assistant does not investigate.",
        }
        answer = (
            "Report this to your manager, People & Culture at hr@innovatech.example.com, "
            "or the Ethics Hotline at ethics@innovatech.example.com. "
            "The company prohibits retaliation for a good-faith report. "
            "I have not investigated this, and I have not stored a ticket."
        )
        steps = []
        if employee_id:
            steps.append(self._ticket_step(employee_id, "workplace_conduct", message))
            steps.append(self._policy_step("harassment discrimination reporting ethics hotline", "Show the conduct policy"))
        return self._finish(message, "conduct", answer, trace, citations, escalation, None, steps)

    def _pto(self, message: str, employee_id: str, trace: list[dict]) -> dict:
        duration = parse_duration(message)
        hours = None if duration.get("days") is None else duration["days"] * 8
        profile = self._call(trace, "lookup_employee_profile", {"employee_id": employee_id})
        if profile.get("error"):
            return self._missing_employee(message, trace, profile)
        balance = self._call(trace, "check_pto_balance", {"employee_id": employee_id, "requested_hours": hours})
        citations = self._search(trace, message + " PTO approval notice WorkDay manager")
        decision = self._call(
            trace,
            "check_policy_compliance",
            {"employee_id": employee_id, "scenario": "pto", "requested_hours": hours},
        )
        employee = profile["employee"]
        available = balance.get("available_hours")
        notice = decision.get("notice_business_days_required")
        decision_name = decision.get("decision")
        if decision_name == "not_eligible":
            verdict = (
                "You are not eligible for PTO. Temporary employees and contractors are not covered "
                "unless a contract says otherwise."
            )
        elif hours is None:
            verdict = f"{self._pto_left_sentence(available)} Tell me how many days you need."
        elif decision_name == "insufficient_balance":
            shortfall = max(0, float(hours) - float(available or 0))
            unpaid = self._search(trace, "5.6 unpaid time off when the PTO balance is not enough")
            matching = [
                item
                for item in unpaid
                if str(item.get("document_id")) == "200" and str(item.get("section") or "").startswith("5.6")
            ]
            if matching:
                citations = matching[:1]
            verdict = (
                f"{self._pto_left_sentence(available)} This request is {hours:.0f} hours, "
                "which exceeds that balance, so you do not have enough paid time saved. "
                f"You can use the {self._number(available)} hours you have and request the remaining "
                f"{self._number(shortfall)} hours as unpaid time off. "
                "If the absence is longer than 4 weeks, or it is for a medical or family reason, "
                "contact People & Culture about a leave of absence."
            )
        else:
            verdict = (
                f"{self._pto_left_sentence(available)} This request is {hours:.0f} hours, so you have enough time."
            )
        if notice and decision_name != "not_eligible":
            verdict += (
                f" You still need to give at least {notice} business days' notice and get approval "
                f"from {employee['manager_name']} in WorkDay."
            )
        verdict += " This is not an approval."
        details = (
            f"I would like to request PTO"
            + (f" for {duration['days']} days ({hours:.0f} hours)." if hours is not None else ".")
            + f" My available balance is {available} hours. "
            "I understand this still needs manager approval in WorkDay."
        )
        policy_label = "Show the unpaid time off policy" if decision_name == "insufficient_balance" else "Show the PTO policy"
        policy_section = "5.6 Unpaid Time Off" if decision_name == "insufficient_balance" else None
        steps = [
            self._policy_step(
                "5.6 unpaid time off" if policy_section else message + " PTO approval notice WorkDay",
                policy_label,
                section=policy_section,
                document_id="200" if policy_section else None,
            ),
            self._email_step(employee_id, "manager", "pto", details, f"Draft an email to {employee['manager_name']}"),
            self._ticket_step(employee_id, "pto", message),
        ]
        return self._finish(message, "pto", verdict, trace, citations, None, None, steps)

    def _remote(self, message: str, employee_id: str, trace: list[dict]) -> dict:
        duration = parse_duration(message)
        profile = self._call(trace, "lookup_employee_profile", {"employee_id": employee_id})
        if profile.get("error"):
            return self._missing_employee(message, trace, profile)
        lowered = message.lower()
        located = (
            duration.get("weeks") is not None
            or "another" in lowered
            or "domestic" in lowered
            or mentions_international(message)
        )
        scenario = "remote_location" if located else "remote_eligibility"
        decision = self._call(
            trace,
            "check_policy_compliance",
            {
                "employee_id": employee_id,
                "scenario": scenario,
                "duration_weeks": duration.get("weeks"),
                "duration_days": duration.get("days"),
                "location_text": message,
            },
        )
        employee = profile["employee"]
        policy_query, section = self._remote_source(decision)
        citations = self._search(trace, policy_query)
        citations += self._search(trace, "VPN GlobalProtect company devices remote work data security")
        if re.search(r"\b(training|course|conference|seminar)\b", message, re.I):
            citations += self._search(
                trace,
                "business travel CorporateTravel manager pre-approval international GSA per diem",
            )
        citations = self._keep_remote_citations(citations, section)
        answer = self._remote_answer(decision, message)
        if self._snippets_contain(citations, "GlobalProtect"):
            answer += " You also need to use the company VPN, GlobalProtect, on a company device."
        answer += " This is not an approval."
        details = " ".join(decision.get("reasons") or []) or message
        recipient = "hr" if decision.get("escalate_to") else "manager"
        recipient_label = "People & Culture" if recipient == "hr" else employee["manager_name"]
        escalation = None
        if decision.get("escalate_to"):
            escalation = {
                "recommended": True,
                "team": " and ".join(decision["escalate_to"]),
                "reason": decision.get("decision"),
            }
        steps = [
            self._policy_step(policy_query, "Show the matching remote-work policy", section=section),
            self._email_step(employee_id, recipient, "remote", details, f"Draft an email to {recipient_label}"),
            self._ticket_step(employee_id, "remote_work", message),
        ]
        return self._finish(message, "remote", answer, trace, citations, escalation, None, steps)

    def _expense(self, message: str, employee_id: str, trace: list[dict]) -> dict:
        profile = self._call(trace, "lookup_employee_profile", {"employee_id": employee_id})
        if profile.get("error"):
            return self._missing_employee(message, trace, profile)
        citations = self._search_topics(trace, message)
        if not citations:
            citations = self._search(trace, message)
        decision = self._call(
            trace,
            "check_policy_compliance",
            {"employee_id": employee_id, "scenario": "expense", "expense_item": message},
        )
        employee = profile["employee"]
        answer = self._expense_answer(decision, message)
        steps = [
            self._policy_step(message, "Show the expense policy"),
            self._email_step(
                employee_id,
                "manager",
                "expense",
                " ".join(decision.get("reasons") or []),
                f"Draft an email to {employee['manager_name']}",
            ),
            self._ticket_step(employee_id, "expense", message),
        ]
        return self._finish(message, "expense", answer, trace, citations, None, None, steps)

    def _benefits(self, message: str, employee_id: str, trace: list[dict]) -> dict:
        self._call(trace, "lookup_benefits_status", {"employee_id": employee_id})
        profile = self._call(trace, "lookup_employee_profile", {"employee_id": employee_id})
        if profile.get("error"):
            return self._missing_employee(message, trace, profile)
        search_text = message + " benefits eligibility"
        if re.search(r"\b(deductible\w*|copay\w*|premium\w*|coinsurance)\b", message, re.I):
            search_text = message + " annual deductible plan comparison"
        citations = self._search(trace, search_text)
        topic = benefits_topic(message)
        decision = self._call(
            trace,
            "check_policy_compliance",
            {"employee_id": employee_id, "scenario": "benefits", "benefits_topic": topic},
        )
        reasons = decision.get("reasons") or []
        answer = self._benefits_answer(decision, message)
        fact = plan_fact(message, decision.get("benefits"))
        if fact and fact not in answer:
            answer = answer.replace(
                " This does not enroll you or change a benefit.",
                f" {fact} This does not enroll you or change a benefit.",
            )
        steps = [
            self._policy_step(message + " benefits eligibility", "Show the benefits policy"),
            self._email_step(employee_id, "hr", "benefits", " ".join(reasons), "Draft an email to People & Culture"),
        ]
        return self._finish(message, "benefits", answer, trace, citations, None, None, steps)

    def _policy_qa(self, message: str, trace: list[dict], intent: str, employee_id: str | None = None) -> dict:
        named = re.search(r"\b(?:policy|document)\s+(\d{3})\b", message, re.I)
        citations: list[dict] = []
        if named:
            section = self._call(
                trace,
                "get_policy_section",
                {"document_id": named.group(1), "section": message},
            )
            citations.extend(section.get("citations") or [])
        search_message = message
        if re.search(r"\b(deductible\w*|copay\w*|premium\w*)\b", message, re.I):
            search_message = f"{message} annual deductible plan comparison"
        citations.extend(self._search_topics(trace, message) or self._search(trace, search_message))
        filler = {
            "can", "how", "what", "many", "much", "long", "someone", "here", "when", "where",
            "which", "does", "did", "year", "years", "employees", "employee", "required", "need",
            "needs", "each", "per", "week", "weeks", "days", "day", "soon", "quickly", "worked",
            "give", "resigning", "written", "notice", "returned", "reported", "submitted",
            "itemized", "must", "should", "company", "have", "been", "your", "our",
            "tell", "work", "about", "steps", "step", "applying", "apply", "procedure", "process",
        }
        absent = [term for term in missing_terms(message) if term not in filler and len(term) >= 3]
        if not citations:
            answer = (
                "I could not find supporting text in the Innovatech policy corpus for that question. "
                "I will not guess. Contact People & Culture at hr@innovatech.example.com."
            )
            return self._finish(message, "out_of_scope", answer, trace, [], None, None)
        fact = plan_fact(message)
        prose = self._to_plain_english(self._direct_policy(citations, message)).strip()
        if fact:
            answer = fact
        else:
            answer = prose
        if absent:
            answer = "The policies do not mention " + ", ".join(absent) + ". " + answer
        if not answer.strip():
            answer = "I could not find a sentence that answers that."
        if employee_id and _applying_for_pto(message):
            left = self._pto_left_for(employee_id, trace)
            if left:
                answer = f"{left} {answer}"
        steps = [self._policy_step(message, "Show the full policy text")]
        return self._finish(message, "policy_qa" if intent == "policy_qa" else intent, answer, trace, citations, None, None, steps)

    def _search_topics(self, trace: list[dict], message: str) -> list[dict]:
        matched = [name for name, pattern in CATEGORIES if pattern.search(message)]
        if len(matched) < 2:
            return []
        citations: list[dict] = []
        for name in matched[:3]:
            citations.extend(self._search(trace, f"{message} {CATEGORY_QUERY[name]}"))
        return citations

    def _search(self, trace: list[dict], query: str, document_id: str | None = None) -> list[dict]:
        arguments: dict = {"query": query, "top_k": retrieval_k()}
        if document_id:
            arguments["document_id"] = document_id
        result = self._call(trace, "search_policy_documents", arguments)
        citations = result.get("citations") or []
        for citation in citations:
            citation["batch"] = len(trace)
        return citations

    def _id_from_lookup(self, trace: list[dict], message: str) -> str | None:
        found = self._call(trace, "lookup_employee_profile", {"name": message})
        if found.get("employee"):
            return str(found["employee"]["employee_id"])
        return None

    def _missing_employee(self, message: str, trace: list[dict], profile: dict) -> dict:
        return self._finish(
            message,
            "clarify",
            profile.get("message") or "I could not find that employee in the synthetic directory. Nothing was changed.",
            trace,
            [],
            None,
            None,
        )

    def _call(self, trace: list[dict], name: str, arguments: dict) -> dict:
        try:
            output = self.mcp.call_tool(name, arguments)
            status = "error" if output.get("error") else "ok"
        except McpError as exc:
            output = {"error": "mcp_unavailable", "detail": str(exc)}
            status = "error"
        trace.append(
            {
                "step": len(trace) + 1,
                "tool": name,
                "arguments": arguments,
                "status": status,
                "output": output,
            }
        )
        return output

    def _content_terms(self, text: str) -> set[str]:
        terms = set()
        for token in tokenize(text):
            if token in STOP or len(token) <= 2:
                continue
            terms.add(token)
            if token.endswith("s") and len(token) > 4:
                terms.add(token[:-1])
        return terms

    def _remote_section_preference(self, question: str) -> str | None:
        """The remote section that matches the length and place, when the question states them."""
        explicit = re.search(r"\b(5\.[123]|3\.0)\b", question or "")
        if explicit:
            return explicit.group(1)
        lowered = (question or "").lower()
        if not re.search(r"remote|domestic|hybrid|another state|work from", lowered):
            return None
        if mentions_international(question or "") and not re.search(r"\bdomestic\b", lowered):
            return "5.2"
        weeks = parse_duration(question or "").get("weeks")
        if weeks is None and _wants_location_rules(question or ""):
            return None
        longer_than_cap = weeks is not None and weeks >= 4 and re.search(
            r"\b(more than|longer than|over|extended)\b", lowered
        )
        if weeks is not None and (weeks > 4 or longer_than_cap):
            return "5.3"
        if weeks is not None and weeks <= 4:
            return "5.1"
        if re.search(r"domestic|temporary remote|different location", lowered):
            return "5.1"
        return None

    def _remote_source(self, decision: dict) -> tuple[str, str]:
        name = decision.get("decision")
        if name == "requires_change_of_work_location":
            return (
                "5.3 Extended Remote Work another U.S. state exceeding 4 weeks Change of Work Location",
                "5.3 Extended Remote Work",
            )
        if name == "escalate_international":
            return (
                "5.2 international remote work outside the United States generally not permitted Legal",
                "5.2 Temporary Remote Work",
            )
        if name == "manager_approval_for_temporary_remote":
            return (
                "5.1 Temporary Remote Work Domestic different location within the United States not exceeding 4 weeks",
                "5.1 Temporary Remote Work",
            )
        return ("3.0 hybrid remote eligibility 3 days per week", "3.0 Eligibility")

    def _keep_remote_citations(self, citations: list[dict], section: str) -> list[dict]:
        """Keep the matching remote section and the VPN section. Drop the other remote sections."""
        prefix = section.split()[0]
        kept = []
        seen = set()
        for item in citations:
            document_id = str(item.get("document_id") or "")
            heading = str(item.get("section") or "")
            if document_id == "300" and heading.startswith(prefix):
                key = (document_id, heading)
            elif document_id == "600" and heading.startswith("3.3"):
                key = (document_id, heading)
            elif document_id == "900":
                key = (document_id, heading)
            else:
                continue
            if key in seen:
                continue
            seen.add(key)
            kept.append(item)
        return kept or citations

    def _choose_citations(self, citations: list[dict], question: str = "", *, include_related: bool = True) -> list[dict]:
        """Best chunk from each search, then the chunks that share the most question terms."""
        question_terms = {token for token in tokenize(question) if token not in STOP and len(token) > 2}
        preference = self._remote_section_preference(question)
        if preference:
            matched = [
                item
                for item in citations
                if str(item.get("document_id")) == "300" and str(item.get("section") or "").startswith(preference)
            ]
            if matched:
                citations = [
                    item
                    for item in citations
                    if str(item.get("document_id")) != "300" or str(item.get("section") or "").startswith(preference)
                ]
        batches: dict[int, list[dict]] = {}
        for citation in citations:
            if str(citation.get("document_id") or "") == "010":
                continue
            batches.setdefault(citation.get("batch", 0), []).append(citation)
        chosen: list[dict] = []
        seen: set[tuple[str, str]] = set()

        def add(citation: dict) -> None:
            key = (citation.get("document_id", ""), citation.get("section", ""))
            if key in seen or not citation:
                return
            seen.add(key)
            chosen.append(citation)

        for batch in batches.values():
            add(max(batch, key=lambda item: item.get("score") or 0))
        if question_terms:
            for batch in batches.values():
                add(
                    max(
                        batch,
                        key=lambda item: (
                            len(question_terms & set(tokenize(item.get("snippet") or ""))),
                            item.get("score") or 0,
                        ),
                    )
                )
        if not include_related:
            return chosen
        leftovers = sorted(
            [item for item in citations if str(item.get("document_id") or "") != "010"],
            key=lambda item: (
                len(question_terms & set(tokenize(item.get("snippet") or ""))),
                item.get("score") or 0,
            ),
            reverse=True,
        )
        for citation in leftovers:
            if len(chosen) >= 4:
                break
            add(citation)
        return chosen

    def _fact_lines(self, text: str) -> list[tuple[str, bool, bool]]:
        lines = []
        for raw in re.split(r"\n+", text or ""):
            original = raw.strip().replace("**", "")
            if re.fullmatch(r"procedure:?", original, re.I):
                continue
            is_step = bool(re.match(r"^\d+\.\s+\S", original))
            is_violation = bool(re.search(r"\bviolat", original, re.I))
            line = re.sub(r"^[-*]\s*", "", original)
            line = re.sub(r"^\d+\.\s*", "", line)
            if not line or set(line) <= set("|:- "):
                continue
            if line.startswith("|"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                cells = [cell for cell in cells if cell and not re.fullmatch(r":?-+:?", cell)]
                if len(cells) < 2:
                    continue
                line = " — ".join(cells)
                is_step = False
            if len(line) < 12:
                continue
            lines.append((line, is_step, is_violation))
        return lines

    def _location_rules(self, message: str, employee_id: str | None, trace: list[dict]) -> dict:
        """Procedure and violation text for working from another location."""
        international = mentions_international(message) and not re.search(r"\bdomestic\b", message, re.I)
        if international:
            citations = self._search(trace, "5.2 international remote work unapproved serious policy violation")
        else:
            citations = self._search(trace, "5.1 Temporary Remote Work Domestic different location within the United States")
            citations += self._search(
                trace,
                "5.3 Change of Work Location procedure WorkDay tax review violation another state",
            )
        answer = _location_rule_answer(message)
        status = self._authorization_for(employee_id, trace)
        if status:
            answer = f"{status} {answer}"
        steps = [self._policy_step(message, "Show the matching remote-work policy")]
        return self._finish(message, "policy_qa", answer, trace, citations, None, None, steps)

    def _direct_policy(self, citations: list[dict], question: str) -> str:
        """The sentences that answer the question, without the rest of the section."""
        terms = self._content_terms(question)
        picked: list[str] = []
        seen: set[str] = set()
        wants_procedure = bool(_PROCEDURE.search(question))
        wants_violation = bool(_VIOLATION.search(question)) or wants_procedure
        best_steps: list[str] = []
        best_violations: list[str] = []
        best_step_score = -1
        for citation in self._choose_citations(citations, question, include_related=wants_procedure or wants_violation)[:4]:
            marked = self._fact_lines(citation.get("snippet") or citation.get("text") or "")
            lines = [line for line, _step, _violation in marked]
            remaining = set(terms)
            pool = list(lines)
            chosen_lines: list[str] = []
            while pool and len(chosen_lines) < 3 and remaining:
                best = max(pool, key=lambda line: len(remaining & self._content_terms(line)))
                overlap = remaining & self._content_terms(best)
                if not overlap:
                    break
                chosen_lines.append(best)
                remaining -= overlap
                pool.remove(best)
            needs_amount = bool(re.search(r"\b(reimburse\w*|stipend|match|limit|hotel|cost|how much)\b", question, re.I))
            wants_form = bool(re.search(r"\b(file|submit)\b", question, re.I))
            has_amount = any(re.search(r"\$\s?\d", line) for line in chosen_lines)
            for line in lines:
                if line in chosen_lines:
                    continue
                phrases = re.findall(r'"([^"]+)"', line)
                if wants_form and any(" " in phrase for phrase in phrases):
                    chosen_lines.append(line)
                elif needs_amount and not has_amount and re.search(r"\$\s?\d", line):
                    chosen_lines.append(line)
                    has_amount = True
            for line in chosen_lines[:4]:
                key = line.lower()
                if key in seen:
                    continue
                seen.add(key)
                picked.append(line)
            if wants_procedure or wants_violation:
                steps = [line for line, is_step, _violation in marked if is_step]
                violations = [line for line, _step, is_violation in marked if is_violation]
                overlap = len(terms & self._content_terms(" ".join(steps + violations)))
                if steps and overlap > best_step_score:
                    best_step_score = overlap
                    best_steps = steps
                    best_violations = violations
                elif violations and overlap > best_step_score:
                    best_step_score = overlap
                    best_violations = violations
        if wants_procedure and best_steps:
            step_keys = {line.lower() for line in best_steps}
            picked = [line for line in picked if line.lower() not in step_keys]
            numbered = []
            for index, line in enumerate(best_steps, start=1):
                seen.add(line.lower())
                numbered.append(f"{index}. {line}")
            picked = numbered + picked
        if wants_violation:
            for line in best_violations:
                key = line.lower()
                if key in seen:
                    continue
                seen.add(key)
                picked.append(line)
        if re.search(r"\b(reimburse\w*|stipend|match|limit|hotel|cost|how much)\b", question, re.I):
            present = set(re.findall(r"\$\s?\d+", " ".join(picked)))
            best_line = ""
            best_key = (-1, 0)
            for citation in citations:
                if str(citation.get("document_id") or "") == "010":
                    continue
                for line, _step, _violation in self._fact_lines(citation.get("snippet") or citation.get("text") or ""):
                    if line.lower() in seen or not (set(re.findall(r"\$\s?\d+", line)) - present):
                        continue
                    overlap = len(terms & self._content_terms(line))
                    key = (overlap, citation.get("score") or 0)
                    if overlap > 0 and key > best_key:
                        best_key = key
                        best_line = line
            if best_line:
                picked.append(best_line)
        return " ".join(picked)

    def _references(self, citations: list[dict]) -> str:
        best: dict[str, tuple[float, dict]] = {}
        for item in citations:
            document_id = str(item.get("document_id") or "")
            if not document_id or document_id == "010":
                continue
            score = item.get("score") or 0
            current = best.get(document_id)
            if current is None or score > current[0]:
                best[document_id] = (score, item)
        parts = []
        for document_id, (_, item) in sorted(best.items(), key=lambda pair: pair[1][0], reverse=True)[:3]:
            section = (item.get("section") or "").strip()
            number = re.match(r"\d+(?:\.\d+)*", section)
            label = number.group(0) if number else section
            if label:
                parts.append(f"(Reference {document_id} / {label})")
        return " ".join(parts)

    def _policy_blocks(self, citations: list[dict], question: str = "") -> str:
        """Full excerpts, used only after the employee asks to see the policy text."""
        blocks = []
        for citation in self._choose_citations(citations, question)[:4]:
            excerpt = (citation.get("snippet") or citation.get("text") or "").strip()
            blocks.append(f"[{citation.get('document_id')} | {citation.get('section')}]\n{excerpt}")
        if not blocks:
            return "No policy excerpt was retrieved."
        return "\n\n".join(blocks)

    def _run_step(self, message: str, employee_id: str | None, step: dict) -> dict:
        trace: list[dict] = []
        step_id = step.get("id")
        employee_id = employee_id or step.get("employee_id")
        if step_id == "show_policy":
            if step.get("section"):
                found = self._call(
                    trace,
                    "get_policy_section",
                    {"document_id": step.get("document_id") or "300", "section": step["section"]},
                )
                citations = found.get("citations") or []
            else:
                citations = self._search(trace, step.get("policy_query") or message)
            answer = "Here is the policy text:\n\n" + self._policy_blocks(citations, step.get("policy_query") or step.get("section") or message)
            return self._finish(message, "show_policy", answer, trace, citations, None, None)
        if step_id == "draft_email":
            if not employee_id:
                return self._finish(message, "clarify", "Sign in before I draft a message. Nothing was sent.", trace, [], None, None)
            draft = self._call(
                trace,
                "draft_hr_email",
                {
                    "employee_id": employee_id,
                    "recipient": step.get("recipient") or "manager",
                    "purpose": step.get("purpose") or "general",
                    "details": step.get("details") or message,
                },
            )
            return self._finish(
                message,
                "draft_email",
                self._format_draft(draft) + "\n\nThis draft has not been sent.",
                trace,
                [],
                None,
                None,
            )
        if step_id == "prepare_ticket":
            if not employee_id:
                return self._finish(message, "clarify", "Sign in before I prepare a ticket. Nothing was stored.", trace, [], None, None)
            profile = self._call(trace, "lookup_employee_profile", {"employee_id": employee_id})
            if profile.get("error"):
                return self._missing_employee(message, trace, profile)
            pending = {
                "tool": "create_mock_hr_ticket",
                "arguments": {
                    "employee_id": employee_id,
                    "category": step.get("category") or "general",
                    "summary": (step.get("summary") or message)[:800],
                    "confirm": False,
                },
            }
            name = profile["employee"]["full_name"]
            answer = (
                f"I can open a mock HR ticket for {name}. Nothing is stored until you confirm.\n\n"
                f"Draft summary: {pending['arguments']['summary']}"
            )
            return self._finish(message, "ticket", answer, trace, [], None, pending)
        return self._finish(message, "clarify", "I don't have that follow-up. Nothing was changed.", trace, [], None, None)

    def _continue_conversation(self, message: str, employee_id: str | None, history: list[dict] | None, action: str) -> dict:
        """Draft, show policy, or prepare a ticket for the topic already in this chat."""
        trace: list[dict] = []
        topic = _prior_topic(history)
        discussed = _last_assistant_text(history)
        if topic and employee_id and other_employee_requested(topic, employee_id):
            return self._private_refusal()
        if action == "email":
            return self._draft_about_topic(message, employee_id, topic, discussed, trace)
        if action == "policy":
            query = topic or message
            citations = self._search(trace, query)
            answer = "Here is the policy text:\n\n" + self._policy_blocks(citations, query)
            return self._finish(message, "show_policy", answer, trace, citations, None, None)
        if not employee_id:
            return self._finish(message, "clarify", "Sign in before I prepare a ticket. Nothing was stored.", trace, [], None, None)
        if not topic:
            return self._finish(
                message,
                "clarify",
                "Tell me what the ticket should be about. Nothing was stored.",
                trace,
                [],
                None,
                None,
            )
        return self._run_step(
            message,
            employee_id,
            self._ticket_step(employee_id, _purpose_from_text(topic), topic),
        )

    def _draft_about_topic(
        self,
        message: str,
        employee_id: str | None,
        topic: str,
        discussed: str,
        trace: list[dict],
    ) -> dict:
        if not employee_id:
            return self._finish(message, "clarify", "Sign in before I draft a message. Nothing was sent.", trace, [], None, None)
        if not topic:
            return self._finish(
                message,
                "clarify",
                "Tell me what the email should be about. Nothing was sent.",
                trace,
                [],
                None,
                None,
            )
        named = _purpose_from_text(message)
        purpose = named if named != "general" else _purpose_from_text(topic)
        subject = message if named != "general" else topic
        if re.search(r"\b(hr|people & culture|people and culture)\b", message, re.I):
            recipient = "hr"
        elif re.search(r"\bmanager\b", message, re.I):
            recipient = "manager"
        elif purpose == "remote" and (
            mentions_international(f"{topic} {discussed}")
            or re.search(
                r"\b(extended|another state|different state|more than 4)\b",
                f"{topic} {discussed}",
                re.I,
            )
        ):
            recipient = "hr"
        else:
            recipient = "manager"
        details = subject
        if discussed and named == "general":
            details = f"{subject}\n\n{discussed}"
        draft = self._call(
            trace,
            "draft_hr_email",
            {
                "employee_id": employee_id,
                "recipient": recipient,
                "purpose": purpose,
                "details": details[:1200],
            },
        )
        return self._finish(
            message,
            "draft_email",
            self._format_draft(draft) + "\n\nThis draft has not been sent.",
            trace,
            [],
            None,
            None,
        )

    def _policy_step(
        self,
        policy_query: str,
        label: str,
        section: str | None = None,
        document_id: str | None = None,
    ) -> dict:
        step = {"id": "show_policy", "label": label, "policy_query": policy_query}
        if section:
            step["section"] = section
        if document_id:
            step["document_id"] = document_id
        return step

    def _email_step(self, employee_id: str, recipient: str, purpose: str, details: str, label: str) -> dict:
        return {
            "id": "draft_email",
            "label": label,
            "employee_id": employee_id,
            "recipient": recipient,
            "purpose": purpose,
            "details": details,
        }

    def _ticket_step(self, employee_id: str, category: str, summary: str) -> dict:
        return {
            "id": "prepare_ticket",
            "label": "Prepare a mock HR ticket",
            "employee_id": employee_id,
            "category": category,
            "summary": summary,
        }

    def _number(self, value) -> str:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:g}"

    def _pto_left_sentence(self, hours) -> str:
        total = float(hours or 0)
        whole = int(total // 8)
        extra = total - whole * 8
        if extra == 0:
            amount = "1 day" if whole == 1 else f"{whole} days"
        elif whole == 0:
            amount = f"{self._number(extra)} hours"
        else:
            hour_label = "hour" if extra == 1 else "hours"
            amount = f"{whole} days and {self._number(extra)} {hour_label}"
        return f"You have {amount} of PTO left ({self._number(total)} hours)."

    def _pto_left_for(self, employee_id: str | None, trace: list[dict]) -> str:
        if not employee_id:
            return ""
        balance = self._call(trace, "check_pto_balance", {"employee_id": employee_id})
        if balance.get("error") or not balance.get("pto_eligible", True):
            return ""
        if balance.get("available_hours") is None:
            return ""
        return self._pto_left_sentence(balance["available_hours"])

    def _authorization_sentence(self, employee: dict) -> str:
        status = (employee.get("remote_work_status") or "On-Site").strip()
        location = (employee.get("work_location") or "your assigned office").strip()
        eligible = bool(employee.get("remote_work_eligibility"))
        lowered = status.lower()
        current = f"Your current work location is {location}, and you are authorized as {status}."
        if lowered == "hybrid" and eligible:
            return (
                f"{current} You can work hybrid, up to 3 remote days per week, without a new approval. "
                f"You can work on-site at {location} without a remote approval. "
                "You cannot work fully remote without approval."
            )
        if lowered == "fully remote" and eligible:
            return (
                f"{current} You can work fully remote without a new approval. "
                f"You can work on-site at {location} without a remote approval. "
                "You cannot change to a hybrid schedule without approval."
            )
        return (
            f"{current} You can work on-site at {location} without a remote approval. "
            "You cannot work remote or hybrid without approval from your manager and department head."
        )

    def _authorization_for(self, employee_id: str | None, trace: list[dict]) -> str:
        if not employee_id:
            return ""
        profile = self._call(trace, "lookup_employee_profile", {"employee_id": employee_id})
        employee = profile.get("employee") or {}
        if profile.get("error") or not employee:
            return ""
        return self._authorization_sentence(employee)

    def _remote_answer(self, decision: dict, message: str = "") -> str:
        name = decision.get("decision")
        if name == "escalate_international":
            return self._international_options(decision, message)
        opening = self._authorization_sentence(decision)
        if name == "requires_change_of_work_location":
            follow = (
                "Working from another U.S. state for more than 4 weeks needs a Change of Work Location request in WorkDay. "
                + _CHANGE_OF_LOCATION_STEPS
                + " "
                + _UNAPPROVED_LOCATION_VIOLATION
            )
        elif name == "not_remote_eligible":
            follow = "Your role is not set up for remote work. Your manager and department head have to approve it first."
        elif name == "manager_approval_for_temporary_remote":
            weeks = decision.get("duration_weeks")
            length = f" for {self._number(weeks)} weeks" if weeks else ""
            follow = (
                f"This is temporary remote work{length}, inside the United States. "
                "It stays within the 4 week limit, so you do not need an extended remote request. "
                "Your manager has to approve it."
            )
        else:
            follow = "If you need to work fully remote, you need to request approval."
        return f"{opening} {follow}"

    def _international_options(self, decision: dict, message: str) -> str:
        location = decision.get("work_location") or "your assigned office"
        status = decision.get("remote_work_status") or "your current status"
        parts = [
            f"Your current work location is {location}, and you are authorized as {status}. "
            "That authorization does not cover work outside the United States.",
            "You have two options, and neither one approves the trip by itself.",
            "Option 1: ask People & Culture and Legal to review an exception. "
            "Working outside the United States is generally not permitted. "
            "Approval is rare and only for extenuating circumstances.",
        ]
        if re.search(r"\b(training|course|conference|seminar)\b", message or "", re.I):
            parts.append(
                "A training course is not listed as an extenuating circumstance, so do not assume it will be approved."
            )
            parts.append(
                "Option 2: travel to the training and then return, without moving your work location. "
                "That is business travel. Your manager has to pre-approve it before you book. "
                "Book the trip through CorporateTravel at least 14 days ahead. "
                "Flights are economy class. Meals outside the United States use the GSA per diem for that location."
            )
        else:
            parts.append(
                "There is no separate approval for simply working from another country."
            )
        weeks = decision.get("duration_weeks")
        if weeks is not None:
            parts.append(
                f"Asking for {self._number(weeks)} weeks does not use the domestic location rule. "
                "The 4-week limit applies only inside the United States."
            )
        if not decision.get("international_remote_approved"):
            parts.append("Your employee record does not show international remote work as approved.")
        parts.append("Working outside the United States without approval is a serious policy violation.")
        return " ".join(parts)

    def _expense_answer(self, decision: dict, message: str) -> str:
        name = decision.get("decision")
        remaining = decision.get("stipend_remaining")
        status = decision.get("remote_work_status") or "your current"
        if name == "not_reimbursable":
            body = "That item is not covered by the home office stipend. Desks, internet service, coffee makers, and general office supplies are not eligible."
        elif name == "company_issued":
            body = "The company issues your laptop. A personal laptop is not part of the stipend. An ergonomic chair can be, from the one-time $500 stipend, if you submit it in Concur."
        elif name == "not_eligible_for_stipend":
            body = f"The one-time $500 home office stipend is only for remote or hybrid employees. You are currently listed as a {status} employee."
        elif name == "stipend_exhausted":
            body = "You already used the one-time $500 home office stipend, so it cannot be used again."
        elif name == "reimbursable_from_stipend":
            item = "an ergonomic chair" if re.search(r"chair", message, re.I) else "this item"
            body = (
                f"You can request reimbursement for {item} from the one-time $500 home office stipend. "
                f"You have ${self._number(remaining)} left. Buy it yourself, then submit the receipt in Concur."
            )
        elif name == "travel_preapproval_required":
            body = (
                "Your manager has to approve the trip before you book. "
                "Hotel rates should not exceed $300 a night, and meals are $75 per day inside the United States."
            )
        else:
            body = "Get manager approval first. Keep an itemized receipt for anything over $25, and submit it in Concur within 30 days."
        return body + " This is not a reimbursement approval."

    def _benefits_answer(self, decision: dict, message: str = "") -> str:
        name = decision.get("decision")
        benefits = decision.get("benefits") or {}
        if name == "not_eligible":
            body = "You are not eligible for company benefits. Temporary and contract employees are not covered."
        elif name == "hsa_eligible":
            enrolled = "You are already enrolled." if benefits.get("hsa_enrolled") else "You are not enrolled yet."
            body = (
                "You can use an HSA because you are on a high-deductible health plan. "
                f"The company contributes $500 a year. {enrolled}"
            )
        elif name == "hsa_not_eligible":
            body = "An HSA is only for a high-deductible health plan. Your plan is different. You may still be able to use an FSA."
        elif name == "tuition_not_eligible":
            body = "Tuition help is only for full-time employees after one year."
        elif name == "tuition_not_yet":
            body = "Tuition help of up to $5,000 a year starts after one year of full-time work."
        elif name == "tuition_eligible":
            body = "You can ask for up to $5,000 a year for job-related courses. Your manager has to approve it first."
        elif name == "retirement_eligible":
            rate = benefits.get("401k_contribution_percent")
            kind = benefits.get("401k_contribution_type")
            kind_text = f" ({kind})" if kind else ""
            body = (
                "The company matches 100% of your 401k contributions up to 4%. "
                f"You are currently contributing {rate}%{kind_text}."
            )
        elif name == "enrolled":
            plan = benefits.get("medical_plan") or "on file"
            medical_cost = re.search(r"\b(deductible\w*|copay\w*|premium\w*|coinsurance)\b", message, re.I)
            extras = [plan_name for plan_name in (benefits.get("dental_plan"), benefits.get("vision_plan")) if plan_name]
            extra = f" You also have {' and '.join(extras)}." if extras and not medical_cost else ""
            rate = benefits.get("401k_contribution_percent")
            if medical_cost:
                retirement = ""
            elif rate:
                kind = benefits.get("401k_contribution_type")
                kind_text = f" ({kind})" if kind else ""
                retirement = f" Your 401k contribution is {rate}%{kind_text}."
            else:
                retirement = " You are not currently contributing to the 401k."
            body = f"You are enrolled in benefits. Your medical plan is {plan}.{extra}{retirement}"
        else:
            body = "You can enroll in benefits, but you are not enrolled yet."
        return body + " This does not enroll you or change a benefit."

    def _to_plain_english(self, text: str) -> str:
        spoken = []
        for piece in re.split(r"(?<=[.!?])\s+|\s{2,}", text or ""):
            line = piece.strip()
            if not line:
                continue
            row = re.match(r"(.+?)\s+[—-]{1,2}\s+(.+?)\s+[—-]{1,2}\s+(.+)", line)
            if row and re.search(r"\d+\s+days", row.group(2), re.I):
                spoken.append(f"Employees with {row.group(1)} of service accrue {row.group(2)} each year.")
                continue
            line = re.sub(r"^Ineligible Items:\s*", "These items are not covered: ", line)
            line = re.sub(r"^Eligible Items:\s*", "These items can be covered: ", line)
            line = re.sub(r"^Procedure:\s*", "", line)
            if line:
                spoken.append(line)
        return " ".join(spoken)

    def _snippets_contain(self, citations: list[dict], term: str) -> bool:
        return any(term.lower() in (item.get("snippet") or "").lower() for item in citations)

    def _format_draft(self, draft: dict) -> str:
        if draft.get("error"):
            return "I could not draft a message because the employee record was unavailable."
        return f"Draft only — not sent\nTo: {draft.get('to')}\nSubject: {draft.get('subject')}\n\n{draft.get('body')}"

    def _finish(self, message, intent, answer, trace, citations, escalation, pending, next_steps=None) -> dict:
        unique = []
        seen = set()
        for citation in citations:
            key = (citation.get("document_id"), citation.get("section"), citation.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(
                {
                    "document_id": citation.get("document_id"),
                    "title": citation.get("title"),
                    "section": citation.get("section"),
                    "source": citation.get("source"),
                    "snippet": citation.get("snippet"),
                    "score": citation.get("score"),
                }
            )
        polished = rewrite(message, answer)
        final = polished or answer
        if citations and intent != "show_policy" and "(Reference " not in final:
            references = self._references(self._choose_citations(citations, message, include_related=False))
            if references:
                final = final.rstrip() + " " + references
        if next_steps and "What would you like to do next?" not in final:
            final = final.rstrip() + "\n\nWhat would you like to do next?"
        return {
            "answer": final,
            "llm_synthesis": polished is not None,
            "intent": intent,
            "citations": unique,
            "snippets": [item.get("snippet") for item in unique if item.get("snippet")],
            "trace": trace,
            "escalation": escalation,
            "pending_action": pending,
            "next_steps": next_steps or [],
            "action_taken": any(
                step.get("tool") == "create_mock_hr_ticket" and (step.get("output") or {}).get("stored")
                for step in trace
            ),
        }

    def _explicit_id(self, message: str) -> str | None:
        match = EMPLOYEE_ID.search(message)
        return match.group(1) if match else None

    def _classify(self, message: str) -> str:
        message = re.sub(r"401\s*\(\s*k\s*\)", "401k", message, flags=re.I)
        if OUT_OF_SCOPE.search(message) and not any(pattern.search(message) for _, pattern in CATEGORIES):
            return "out_of_scope"
        if SENSITIVE.search(message):
            return "conduct"
        if TICKET.search(message):
            return "ticket"
        matched = [name for name, pattern in CATEGORIES if pattern.search(message)]
        if not matched:
            if _corpus_covers(message):
                return "policy_qa"
            return "out_of_scope"
        primary = matched[0]
        personal = PERSONAL.search(message) is not None
        if primary == "pto" and personal:
            if parse_duration(message).get("days") is None and not re.search(r"\b(balance|accrual|how much)\b", message, re.I):
                return "clarify_pto"
            return "pto"
        if primary == "remote" and personal:
            if parse_duration(message).get("weeks") is None and not re.search(
                r"another state|other state|different state|work from|domestic", message, re.I
            ) and not mentions_international(message):
                return "clarify_remote"
            return "remote"
        if primary == "expense" and personal:
            return "expense"
        if primary == "benefits" and personal:
            return "benefits"
        if "equipment" in matched and re.search(r"\breturn\b", message, re.I):
            return "policy_qa"
        return "policy_qa"


_PROCEDURE = re.compile(r"\b(procedure|steps|process|how (?:do i|to)|apply for)\b", re.I)
_VIOLATION = re.compile(r"\b(violation|without approval|unapproved|not reported|disciplinary)\b", re.I)
_CHANGE_OF_LOCATION_STEPS = (
    "1. You submit a Change of Work Location request in WorkDay. "
    "2. Your manager approves the request. "
    "3. People & Culture and Finance conduct a tax review. "
    "4. People & Culture gives the final approval."
)
_UNAPPROVED_LOCATION_VIOLATION = (
    "Working from another state that is not reported and approved is a violation of policy "
    "and may lead to disciplinary action."
)


def _applying_for_pto(message: str) -> bool:
    if not re.search(r"\b(pto|vacation|time off)\b", message or "", re.I):
        return False
    return bool(_PROCEDURE.search(message or "") or re.search(r"\b(apply|request(?:ing)?)\b", message or "", re.I))


def _wants_location_rules(message: str) -> bool:
    """A question about the steps or the violation for working somewhere else."""
    if not re.search(r"\b(remote\w*|work(?:ing)? from|different location|another location|another state|other state|different state)\b", message or "", re.I) and not mentions_international(message or ""):
        return False
    if not (_PROCEDURE.search(message or "") or _VIOLATION.search(message or "")):
        return False
    return bool(
        mentions_international(message or "")
        or re.search(
            r"\b(different location|another location|different state|another (?:u\.?s\.? )?state|other state|work location|domestic|international|abroad|location)\b",
            message or "",
            re.I,
        )
    )


def _location_rule_answer(message: str) -> str:
    international = mentions_international(message) and not re.search(r"\bdomestic\b", message, re.I)
    if international:
        return (
            "Working outside the United States is generally not permitted. "
            "Any request has to go to People & Culture and Legal, and approval is rare. "
            "Unapproved international work is a serious policy violation. "
            "You can ask People & Culture and Legal to review an exception. "
            "Approval is rare and only for extenuating circumstances. "
            "This is not an approval. (Reference 300 / 5.2)"
        )
    weeks = parse_duration(message).get("weeks")
    extended = bool(re.search(r"\bextended\b", message, re.I))
    parts = []
    refs = []
    if not extended and (weeks is None or weeks <= 4):
        parts.append(
            "For up to 4 weeks inside the United States, your manager has to approve working from the different location. "
            "The request is also subject to business needs and a tax and legal review."
        )
        refs.append("(Reference 300 / 5.1)")
    if extended or weeks is None or weeks > 4 or _PROCEDURE.search(message):
        parts.append(
            "For more than 4 weeks in another U.S. state, file a Change of Work Location. "
            + _CHANGE_OF_LOCATION_STEPS
        )
        refs.append("(Reference 300 / 5.3)")
    parts.append(_UNAPPROVED_LOCATION_VIOLATION)
    if "(Reference 300 / 5.3)" not in refs:
        refs.append("(Reference 300 / 5.3)")
    return " ".join(parts) + " This is not an approval. " + " ".join(refs)


def parse_duration(text: str) -> dict:
    lowered = text.lower()
    weeks = _number_before(lowered, "week")
    days = _number_before(lowered, "day")
    if weeks is None and re.search(r"\b(a|one) week\b", lowered):
        weeks = 1
    if days is None and weeks is not None and re.search(r"\bpto|vacation|time off\b", lowered):
        days = weeks * 5
    elif days is None and weeks is not None:
        days = weeks * 7
    return {"weeks": weeks, "days": days}


def _number_before(text: str, unit: str) -> float | None:
    matches = list(re.finditer(rf"\b(\d+(?:\.\d+)?|{'|'.join(NUMBERS)})\s+{unit}s?\b", text))
    if not matches:
        return None
    raw = matches[-1].group(1)
    return float(raw) if raw[:1].isdigit() else float(NUMBERS[raw])


_FOLLOW_UP = re.compile(
    r"^\s*(and|also|same|instead|rather|what about|how about)\b|\b(what|how) about\b",
    re.I,
)
_EMAIL_REQUEST = re.compile(
    r"\b(draft|write|compose|prepare)\b.{0,40}\b(an? )?(email|e-mail|message)\b"
    r"|\b(email|e-mail)\b.{0,30}\b(manager|hr)\b",
    re.I,
)
_POLICY_REQUEST = re.compile(
    r"\b(show|see|read|display)\b.{0,30}\b(policy|policies|section)\b|\b(full policy|policy text)\b",
    re.I,
)
_TICKET_REQUEST = re.compile(
    r"\b(prepare|create|open|file|submit)\b.{0,40}\b(mock )?(hr )?(ticket|case)\b",
    re.I,
)


def _conversation_action(message: str) -> str | None:
    """A request to act on the current chat, rather than a new policy question."""
    if _EMAIL_REQUEST.search(message or ""):
        return "email"
    if _POLICY_REQUEST.search(message or ""):
        return "policy"
    if _TICKET_REQUEST.search(message or ""):
        return "ticket"
    return None


def _purpose_from_text(text: str) -> str:
    matched = [name for name, pattern in CATEGORIES if pattern.search(text or "")]
    if not matched:
        return "general"
    primary = matched[0]
    if primary == "travel":
        return "expense"
    if primary in {"pto", "remote", "expense", "benefits", "conduct"}:
        return primary
    return "general"


def _prior_topic(history: list[dict] | None) -> str:
    users = [turn["text"] for turn in _clean_history(history) if turn["role"] == "user"]
    for text in reversed(users):
        if _conversation_action(text):
            continue
        return text
    return ""


def _last_assistant_text(history: list[dict] | None) -> str:
    for turn in reversed(_clean_history(history)):
        if turn["role"] != "assistant":
            continue
        text = re.sub(r"\n+What would you like to do next\?\s*$", "", turn["text"]).strip()
        return text
    return ""


def _is_follow_up(message: str) -> bool:
    if _FOLLOW_UP.search(message or ""):
        return True
    return len((message or "").split()) <= 4


def _has_own_topic(message: str) -> bool:
    if OUT_OF_SCOPE.search(message or ""):
        return True
    return any(pattern.search(message or "") for _, pattern in CATEGORIES)


def _clean_history(history: list[dict] | None) -> list[dict]:
    cleaned = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        text = str(item.get("text") or "").strip()
        if role not in {"user", "assistant"} or not text:
            continue
        cleaned.append({"role": role, "text": text[:500]})
    return cleaned[-6:]


def _with_history(message: str, history: list[dict] | None) -> str:
    """Keep the previous question when this turn only adds a detail."""
    if not message or not _is_follow_up(message):
        return message
    users = [turn["text"] for turn in _clean_history(history) if turn["role"] == "user"]
    if not users:
        return message
    start = len(users) - 1
    for index in range(len(users) - 1, -1, -1):
        if _has_own_topic(users[index]) and not _is_follow_up(users[index]):
            start = index
            break
    parts = users[start:] + [message]
    combined = []
    for part in parts:
        if not combined or combined[-1] != part:
            combined.append(part)
    return " ".join(combined)


_TOPIC_FILLER = {
    "can", "how", "what", "many", "much", "long", "someone", "here", "when", "where",
    "which", "does", "did", "year", "years", "tell", "work", "about", "have", "been",
}


def _corpus_covers(message: str) -> bool:
    """True when the question uses words that appear in the policy index."""
    absent = set(missing_terms(message))
    terms = [
        token
        for token in tokenize(message)
        if token not in STOP and token not in _TOPIC_FILLER and len(token) >= 3
    ]
    return any(token not in absent for token in terms)


def benefits_topic(message: str) -> str:
    lowered = message.lower()
    if "hsa" in lowered:
        return "hsa"
    if "tuition" in lowered:
        return "tuition"
    if "401" in lowered:
        return "401k"
    if "fsa" in lowered:
        return "fsa"
    return "general"
