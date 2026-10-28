def sign_in(client, employee_id="101"):
    from app.auth import load_passwords

    response = client.post(
        "/login",
        json={"employee_id": employee_id, "password": load_passwords()[employee_id]},
    )
    assert response.status_code == 200
    return response


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mcp"] == "connected"
    assert body["mcp_tools"] >= 5
    assert body["index_chunks"] > 0


def test_chat_requires_login(client):
    client.post("/logout")
    response = client.post("/chat", json={"message": "How much PTO do I have?"})
    assert response.status_code == 401


def test_login_rejects_a_wrong_password(client):
    response = client.post("/login", json={"employee_id": "101", "password": "not-the-password"})
    assert response.status_code == 401


def test_pto_demo_uses_mcp_tools(client):
    sign_in(client, "101")
    response = client.post(
        "/chat",
        json={"message": "Can I take three days of PTO next week?", "employee_id": "102"},
    )
    body = response.json()
    tools = [step["tool"] for step in body["trace"]]
    assert "lookup_employee_profile" in tools
    assert "check_pto_balance" in tools
    assert "search_policy_documents" in tools
    assert "draft_hr_email" not in tools
    assert "create_mock_hr_ticket" not in tools
    assert any(step["id"] == "draft_email" for step in body["next_steps"])
    draft = next(step for step in body["next_steps"] if step["id"] == "draft_email")
    follow = client.post("/chat", json={"message": draft["label"], "selected_step": draft}).json()
    assert "draft_hr_email" in [step["tool"] for step in follow["trace"]]
    assert "not been sent" in follow["answer"]
    assert "85" in body["answer"]
    assert "5 business days" in body["answer"]
    assert any(item["document_id"] == "200" for item in body["citations"])
    assert body["action_taken"] is False


def test_remote_demo_cites_multiple_policies(client):
    sign_in(client, "101")
    response = client.post(
        "/chat",
        json={
            "message": "Can I work remotely from another U.S. state for six weeks?",
            "employee_id": "101",
        },
    )
    body = response.json()
    documents = {item["document_id"] for item in body["citations"]}
    assert {"300", "600"} <= documents
    assert "Change of Work Location" in body["answer"]
    assert "4 weeks" in body["answer"]
    assert body["escalation"]["recommended"] is True
    assert body["action_taken"] is False


def test_thirty_days_off_uses_the_employee_balance(client):
    sign_in(client, "113")
    body = client.post(
        "/chat",
        json={"message": "I need to take 30 days off. What are my options?"},
    ).json()
    answer = body["answer"]
    assert body["intent"] == "pto"
    assert "15 days of PTO left (120 hours)" in answer
    assert "30 days (240 hours)" in answer
    assert "15 days (120 hours)" in answer
    assert "2. Take all 30 days: 15 days paid and 15 days unpaid." in answer
    assert "Leave of Absence" in answer
    assert "longer than 4 weeks" in answer
    assert "not ordinary unpaid time off" in answer
    assert "Karen Owens" in answer
    assert "bereavement" not in answer.lower()
    assert "5.3" not in answer
    assert "(Reference 200 / 5.5)" in answer
    assert "(Reference 200 / 5.6)" in answer


def test_insufficient_pto_offers_unpaid_time(client):
    sign_in(client, "101")
    body = client.post(
        "/chat",
        json={"message": "Can I take 20 days of PTO next month?"},
    ).json()
    assert "exceeds" in body["answer"]
    assert "unpaid time off" in body["answer"]
    assert "(Reference 200 / 5.6)" in body["answer"]
    assert "5.6" in body["citations"][0]["section"]


def test_out_of_state_work_location_states_approval_requirements(client):
    sign_in(client, "101")
    body = client.post(
        "/chat",
        json={"message": "I need an out of state work location request."},
    ).json()
    answer = body["answer"]
    assert "San Francisco, CA" in answer
    assert "manager has to approve" in answer
    assert "tax and legal review" in answer
    assert "documented and approved by your manager" in answer
    assert "The approval requirements are:" in answer
    assert "1. You submit a Change of Work Location request in WorkDay." in answer
    assert "2. Your manager approves the request." in answer
    assert "3. People & Culture and Finance conduct a tax review." in answer
    assert "4. People & Culture gives the final approval." in answer
    assert "violation of policy" in answer
    assert "(Reference 300 / 5.1)" in answer
    assert "(Reference 300 / 5.3)" in answer
    follow = client.post(
        "/chat",
        json={
            "message": "What are the approval requirements?",
            "history": [
                {"role": "user", "text": "I need an out of state work location request."},
                {"role": "assistant", "text": answer[:500]},
            ],
        },
    ).json()
    assert "Change of Work Location" in follow["answer"]
    assert "tax and legal review" in follow["answer"]
    assert "monitor" not in follow["answer"].lower()
    assert "(Reference 800" not in follow["answer"]


def test_four_weeks_in_south_dakota_states_approval_requirements(client):
    sign_in(client, "103")
    body = client.post(
        "/chat",
        json={"message": "How about if I want to work from South Dakota for 4 weeks?"},
    ).json()
    answer = body["answer"]
    assert body["intent"] == "remote"
    assert "Austin, TX" in answer
    assert "Fully Remote" in answer
    assert "does not cover South Dakota" in answer
    assert "temporary remote work" in answer.lower()
    assert "within the 4 week limit" in answer
    assert "The approval requirements are: Ivy Lopez has to approve it" in answer
    assert "tax and legal review" in answer
    assert "20 temporary remote days left this year (0 of 20 used)" in answer
    assert "4 weeks is 20 business days" in answer
    assert "uses that full allowance" in answer
    assert "Change of Work Location" not in answer
    assert "5.3" not in answer
    assert "(Reference 300 / 5.1)" in answer


def test_two_week_domestic_remote_uses_temporary_section(client):
    sign_in(client, "101")
    body = client.post(
        "/chat",
        json={"message": "Can I work remotely from another state for 2 weeks?"},
    ).json()
    assert "temporary remote work" in body["answer"].lower()
    assert "(Reference 300 / 5.1)" in body["answer"]
    assert "5.3" not in body["answer"]
    assert "Change of Work Location" not in body["answer"]
    shown = next(step for step in body["next_steps"] if step["id"] == "show_policy")
    follow = client.post("/chat", json={"message": shown["label"], "selected_step": shown}).json()
    assert any(item["section"].startswith("5.1") for item in follow["citations"])
    assert not any(item["section"].startswith("5.3") for item in follow["citations"])


def test_remote_application_states_current_authorization(client):
    sign_in(client, "101")
    body = client.post(
        "/chat",
        json={"message": "What is the procedure to apply for remote work from a different location?"},
    ).json()
    answer = body["answer"]
    assert "San Francisco, CA" in answer
    assert "authorized as Hybrid" in answer
    assert "without a new approval" in answer
    assert "on-site" in answer
    assert "fully remote" in answer.lower()
    assert "Change of Work Location" in answer


def test_pto_application_states_days_left(client):
    sign_in(client, "101")
    request = client.post("/chat", json={"message": "Can I take three days of PTO next week?"}).json()
    assert "10 days and 5 hours of PTO left" in request["answer"]
    assert "85" in request["answer"]
    procedure = client.post("/chat", json={"message": "What is the procedure to request PTO?"}).json()
    assert "10 days and 5 hours of PTO left" in procedure["answer"]
    assert "85" in procedure["answer"]
    assert "WorkDay" in procedure["answer"]


def test_extended_remote_steps_are_not_called_missing(client):
    sign_in(client, "101")
    body = client.post(
        "/chat",
        json={"message": "What are the steps for applying for extended remote work from a different state?"},
    ).json()
    answer = body["answer"]
    assert "do not mention" not in answer
    assert "Change of Work Location" in answer
    assert "1. You submit a Change of Work Location request in WorkDay." in answer
    assert "violation of policy" in answer
    assert "(Reference 300 / 5.3)" in answer
    assert "(Reference 300 / 5.1)" not in answer


def test_draft_email_stays_on_the_remote_topic(client):
    sign_in(client, "101")
    body = client.post(
        "/chat",
        json={
            "message": "Can you draft an email to my manager about this?",
            "history": [
                {
                    "role": "user",
                    "text": "What are the steps for applying for extended remote work from a different state?",
                },
                {
                    "role": "assistant",
                    "text": "For more than 4 weeks in another U.S. state, file a Change of Work Location.",
                },
            ],
        },
    ).json()
    answer = body["answer"]
    assert body["intent"] == "draft_email"
    assert "not been sent" in answer
    assert "Remote location request" in answer
    assert "different state" in answer
    assert "Change of Work Location" in answer
    assert "Tell me where you want to work" not in answer


def test_a_country_outside_the_us_is_international(client):
    sign_in(client, "101")
    france = client.post(
        "/chat",
        json={"message": "Can I work remotely from France for two weeks?"},
    ).json()
    assert "generally not permitted" in france["answer"]
    assert "Legal" in france["answer"]
    assert "serious policy violation" in france["answer"]
    assert "(Reference 300 / 5.2)" in france["answer"]
    assert "temporary remote work" not in france["answer"].lower()
    japan = client.post("/chat", json={"message": "Can I work remotely from Japan?"}).json()
    assert japan["intent"] == "remote"
    assert "(Reference 300 / 5.2)" in japan["answer"]
    assert "Tell me where you want to work" not in japan["answer"]
    domestic = client.post(
        "/chat",
        json={"message": "Can I work remotely from New Mexico for two weeks?"},
    ).json()
    assert "(Reference 300 / 5.1)" in domestic["answer"]
    assert "5.2" not in domestic["answer"]
    indiana = client.post(
        "/chat",
        json={"message": "Can I work remotely from Indiana for two weeks?"},
    ).json()
    assert "(Reference 300 / 5.1)" in indiana["answer"]
    assert "5.2" not in indiana["answer"]


def test_international_training_lists_both_options(client):
    sign_in(client, "103")
    body = client.post(
        "/chat",
        json={
            "message": "I'd like to work from internationally for 4 weeks so that I can attend a training course. What are my options?",
        },
    ).json()
    answer = body["answer"]
    assert "Austin, TX" in answer
    assert "generally not permitted" in answer
    assert "People & Culture and Legal" in answer
    assert "extenuating circumstances" in answer
    assert "CorporateTravel" in answer
    assert "GSA" in answer
    assert "serious policy violation" in answer
    assert "(Reference 300 / 5.2)" in answer
    assert "(Reference 900 /" in answer
    assert "Change of Work Location" not in answer
    assert "temporary remote work" not in answer.lower()


def test_conduct_ticket_asks_for_the_coworker_and_incident(client):
    from app.data_store import list_tickets

    sign_in(client, "101")
    before = len(list_tickets())
    report = "I'm bullied by a co-worker. I need help."
    first = client.post("/chat", json={"message": report}).json()
    step = next(item for item in first["next_steps"] if item["id"] == "prepare_ticket")
    asked = client.post("/chat", json={"message": step["label"], "selected_step": step}).json()
    assert asked["pending_action"] is None
    assert asked["action_taken"] is False
    assert "incident details, including the name of the co-worker" in asked["answer"]
    assert "Nothing is stored yet." in asked["answer"]
    assert len(list_tickets()) == before

    missing_name = client.post(
        "/chat",
        json={
            "message": "He yelled at me yesterday in the standup.",
            "history": [
                {"role": "user", "text": report},
                {"role": "assistant", "text": first["answer"]},
                {"role": "user", "text": step["label"]},
                {"role": "assistant", "text": asked["answer"]},
            ],
        },
    ).json()
    assert missing_name["pending_action"] is None
    assert "co-worker's name" in missing_name["answer"]
    assert "So far:" in missing_name["answer"]

    named = client.post(
        "/chat",
        json={
            "message": "Jordan Lee",
            "history": [
                {"role": "user", "text": report},
                {"role": "assistant", "text": first["answer"]},
                {"role": "user", "text": step["label"]},
                {"role": "assistant", "text": missing_name["answer"]},
            ],
        },
    ).json()
    assert named["intent"] != "private"
    pending = named["pending_action"]
    assert pending["tool"] == "create_mock_hr_ticket"
    assert pending["arguments"]["category"] == "workplace_conduct"
    assert "Jordan Lee" in pending["arguments"]["summary"]
    assert "yelled" in pending["arguments"]["summary"]
    assert report in pending["arguments"]["summary"]
    assert named["action_taken"] is False
    assert len(list_tickets()) == before

    coworker = client.post(
        "/chat",
        json={
            "message": "Bob Williams yelled at me during Monday's standup.",
            "history": [
                {"role": "user", "text": report},
                {"role": "assistant", "text": first["answer"]},
                {"role": "user", "text": step["label"]},
                {"role": "assistant", "text": asked["answer"]},
            ],
        },
    ).json()
    assert coworker["intent"] != "private"
    assert "Bob Williams" in coworker["pending_action"]["arguments"]["summary"]
    assert "42" not in coworker["answer"]
    assert len(list_tickets()) == before


def test_inappropriate_touching_is_a_conduct_report(client):
    sign_in(client, "101")
    body = client.post(
        "/chat",
        json={"message": "My co-worker Arthur touched me inappropriately."},
    ).json()
    answer = body["answer"]
    assert body["intent"] == "conduct"
    assert "outside the Innovatech policy corpus" not in answer
    assert "Unwanted physical contact is harassment and is not tolerated" in answer
    assert "ethics@innovatech.example.com" in answer
    assert "Section 4.0" in answer
    assert "not required to report it to the person accused" in answer
    assert "not investigated" in answer
    assert any(item["document_id"] == "100" for item in body["citations"])
    named = client.post(
        "/chat",
        json={"message": "My co-worker Bob Williams touched me inappropriately."},
    ).json()
    assert named["intent"] == "conduct"
    assert "42" not in named["answer"]


def test_bullying_is_a_conduct_report(client):
    sign_in(client, "101")
    body = client.post(
        "/chat",
        json={"message": "I'm bullied by a co-worker. I need help."},
    ).json()
    answer = body["answer"]
    assert body["intent"] == "conduct"
    assert "do not mention" not in answer
    assert "401" not in answer
    assert "Fidelity" not in answer
    assert "Bullying is not tolerated" in answer
    assert "ethics@innovatech.example.com" in answer
    assert "not required to report it to the person accused" in answer
    assert "Section 4.0" in answer
    assert "does not decide guilt or discipline" in answer
    assert "retaliation" in answer
    assert "investigated" in answer
    assert "not triaged" in answer
    assert any(item["document_id"] == "100" for item in body["citations"])
    sections = {item["section"] for item in body["citations"] if item["document_id"] == "100"}
    assert any(section.startswith("3.3") or section.startswith("4.") for section in sections)
    assert body["escalation"]["recommended"] is True
    danger = client.post(
        "/chat",
        json={"message": "There is workplace violence and I am in immediate danger."},
    ).json()
    assert danger["intent"] == "conduct"
    assert "emergency services" in danger["answer"]
    assert "Section 4.0" in danger["answer"]
    related = client.post(
        "/chat",
        json={"message": "Which policies are related to the code of conduct?"},
    ).json()
    related_answer = related["answer"]
    for policy_id in ("200", "300", "510", "600", "700", "900"):
        assert policy_id in related_answer


def test_harassment_ticket_explains_what_happens_next(client):
    sign_in(client, "101")
    incident = (
        "I was walking by the coffee room and Janet made sexual remarks towards me "
        "and when I try to walk away, she was confrontational."
    )
    history = [
        {"role": "user", "text": "I'd like to report a bullying incident"},
        {"role": "assistant", "text": "Before I draft the mock HR ticket, I need the incident details, including the name of the co-worker."},
        {"role": "user", "text": incident},
        {"role": "assistant", "text": "Draft summary: I'd like to report a bullying incident\n\nReported details: " + incident},
        {"role": "user", "text": "Yes, store the mock ticket."},
        {"role": "assistant", "text": "Mock ticket TICK-0004 is stored with status open_mock. People & Culture still has to review it."},
    ]
    body = client.post("/chat", json={"message": "What happens now?", "history": history}).json()
    answer = body["answer"]
    assert body["intent"] == "conduct"
    assert "do not mention" not in answer
    assert "Section 4.0" in answer
    assert "24 to 48 hours" in answer
    assert "Level 3" in answer
    assert "formal internal investigation" in answer
    assert "coaching alone" in answer
    assert "preserve relevant email and chat" in answer
    assert "person accused" in answer
    assert "retaliation" in answer
    assert "not investigated" in answer
    assert "(Reference 100 / 4.5)" in answer
    assert "(Reference 100 / 4.7)" in answer
    assert body["escalation"]["recommended"] is True
    stored = client.post(
        "/chat",
        json={
            "message": "Yes, store the mock ticket.",
            "confirm_action": True,
            "pending_action": {
                "tool": "create_mock_hr_ticket",
                "arguments": {
                    "employee_id": "101",
                    "category": "workplace_conduct",
                    "summary": "I'd like to report a bullying incident\n\nReported details: " + incident,
                    "confirm": False,
                },
            },
        },
    ).json()
    assert stored["action_taken"] is True
    assert "Level 3" in stored["answer"]
    assert "24 to 48 hours" in stored["answer"]
    unrelated = client.post(
        "/chat",
        json={
            "message": "What happens now?",
            "history": [
                {"role": "user", "text": "Can I take 3 days of PTO next week?"},
                {"role": "assistant", "text": "You have 10 days and 5 hours of PTO left (85 hours)."},
            ],
        },
    ).json()
    assert "Level 3" not in unrelated["answer"]
    assert "sexual" not in unrelated["answer"].lower()


def test_domestic_location_procedure_includes_steps_and_violation(client):
    sign_in(client, "101")
    body = client.post(
        "/chat",
        json={"message": "What is the procedure to apply for remote work from a different location?"},
    ).json()
    answer = body["answer"]
    assert "Change of Work Location" in answer
    assert "WorkDay" in answer
    assert "tax review" in answer
    assert "violation of policy" in answer
    assert "disciplinary action" in answer
    assert "(Reference 300 / 5.3)" in answer
    assert "(Reference 300 / 5.1)" in answer
    violation = client.post(
        "/chat",
        json={"message": "What are the policy violations for working from another state without approval?"},
    ).json()
    assert "violation of policy" in violation["answer"]
    assert "disciplinary action" in violation["answer"]
    assert "(Reference 300 / 5.3)" in violation["answer"]


def test_new_healthcare_and_retirement_questions_are_answered(client):
    sign_in(client, "101")
    deductible = client.post("/chat", json={"message": "What is the deductible for PPO Gold?"}).json()
    assert deductible["intent"] != "out_of_scope"
    assert "$500" in deductible["answer"]
    assert "outside the Innovatech policy corpus" not in deductible["answer"]
    vesting = client.post("/chat", json={"message": "How does 401(k) vesting work?"}).json()
    assert vesting["intent"] != "out_of_scope"
    assert "100%" in vesting["answer"]
    mine = client.post("/chat", json={"message": "What is my deductible?"}).json()
    assert "$500" in mine["answer"]
    assert "401" not in mine["answer"]
    assert "401" not in deductible["answer"]


def test_follow_up_keeps_the_earlier_question(client):
    sign_in(client, "101")
    silver = client.post(
        "/chat",
        json={
            "message": "What about PPO Silver?",
            "history": [
                {"role": "user", "text": "What is the deductible for PPO Gold?"},
                {"role": "assistant", "text": "PPO Gold has an individual deductible of $500."},
            ],
        },
    ).json()
    assert "$1,000" in silver["answer"]
    assert "$500" not in silver["answer"]
    fresh = client.post(
        "/chat",
        json={
            "message": "What is the company 401k match?",
            "history": [
                {"role": "user", "text": "What is the deductible for PPO Gold?"},
            ],
        },
    ).json()
    assert "100%" in fresh["answer"]
    assert "$500" not in fresh["answer"]


def test_manager_approval_follow_up_stays_on_pto(client):
    sign_in(client, "113")
    body = client.post(
        "/chat",
        json={
            "message": "What are the manager approval requirements?",
            "history": [
                {"role": "user", "text": "I need to take 30 days off. What are my options?"},
                {
                    "role": "assistant",
                    "text": (
                        "You have 15 days of PTO left (120 hours). "
                        "2. Take all 30 days: 15 days paid and 15 days unpaid. "
                        "The PTO-only option needs at least 10 business days' notice and approval from Karen Owens in WorkDay."
                    ),
                },
            ],
        },
    ).json()
    answer = body["answer"]
    assert body["intent"] == "pto"
    assert "monitor" not in answer.lower()
    assert "keyboard" not in answer.lower()
    assert "(Reference 800" not in answer
    assert "WorkDay" in answer
    assert "Karen Owens" in answer
    assert "10 business days" in answer
    assert "5 business days" in answer
    assert "15 days paid and 15 days unpaid" in answer
    assert "(Reference 200 / 3.3)" in answer or "(Reference 200 / 3.2)" in answer
    fresh = client.post(
        "/chat",
        json={"message": "What are the manager approval requirements?"},
    ).json()
    assert "15 days paid" not in fresh["answer"]


def test_out_of_scope_and_clarification(client):
    sign_in(client, "101")
    weather = client.post("/chat", json={"message": "What is the weather in Paris tomorrow?"}).json()
    assert weather["intent"] == "out_of_scope"
    assert weather["trace"] == [] or all(step.get("tool") is None for step in weather["trace"])
    vague = client.post("/chat", json={"message": "Can I take time off?"}).json()
    assert vague["intent"] == "clarify"
    assert "days" in vague["answer"].lower()


def test_ticket_requires_confirmation(client):
    from app.data_store import list_tickets

    sign_in(client, "101")
    before = len(list_tickets())
    first = client.post(
        "/chat",
        json={"message": "Create an HR ticket for employee 101 about a PTO question.", "employee_id": "101"},
    ).json()
    assert first["pending_action"]["tool"] == "create_mock_hr_ticket"
    assert first["action_taken"] is False
    assert len(list_tickets()) == before
    second = client.post(
        "/chat",
        json={
            "message": "Yes, store the mock ticket.",
            "employee_id": "101",
            "confirm_action": True,
            "pending_action": first["pending_action"],
        },
    ).json()
    assert second["action_taken"] is True
    assert len(list_tickets()) == before + 1


def test_policy_answer_is_a_short_citation(client):
    sign_in(client, "101")
    body = client.post(
        "/chat",
        json={"message": "How many PTO days and hours do employees with 0-2 years of service accrue each year?"},
    ).json()
    assert "15 days" in body["answer"]
    assert "120 hours" in body["answer"]
    assert "(Reference 200 /" in body["answer"]
    assert "What would you like to do next?" in body["answer"]
    assert "[200 |" not in body["answer"]
    assert len(body["answer"]) < 900


def test_signed_in_employee_cannot_read_someone_else(client):
    sign_in(client, "101")
    body = client.post("/chat", json={"message": "What is Bob Williams's PTO balance?"}).json()
    assert body["intent"] == "private"
    assert body["trace"] == []
    assert "42" not in body["answer"]
