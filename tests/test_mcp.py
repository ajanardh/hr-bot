from app.mcp_client import McpClient


def test_mcp_discovers_and_calls_tools():
    client = McpClient()
    client.start()
    try:
        names = {tool["name"] for tool in client.tools}
        assert names >= {
            "search_policy_documents",
            "get_policy_section",
            "lookup_employee_profile",
            "check_pto_balance",
            "lookup_benefits_status",
            "check_policy_compliance",
            "draft_hr_email",
            "create_mock_hr_ticket",
        }
        result = client.call_tool("search_policy_documents", {"query": "parental leave", "top_k": 3})
        assert result["citations"]
        assert any(item["document_id"] == "200" for item in result["citations"])
        balance = client.call_tool("check_pto_balance", {"employee_id": "101", "requested_hours": 24})
        assert balance["sufficient_balance"] is True
        assert balance["available_hours"] == 85
        missing = client.call_tool("lookup_employee_profile", {"employee_id": "999"})
        assert missing["error"] == "employee_not_found"
        blocked = client.call_tool(
            "create_mock_hr_ticket",
            {"employee_id": "101", "category": "general", "summary": "test", "confirm": False},
        )
        assert blocked["stored"] is False
    finally:
        client.close()
