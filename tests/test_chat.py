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
