"""Newline-delimited JSON-RPC MCP server (protocol 2024-11-05, stdio transport).

Stdout is reserved for protocol messages. Logs go to stderr.
"""

from __future__ import annotations

import json
import sys

from app.tools import TOOL_SPECS, call_tool

PROTOCOL = "2024-11-05"


def _respond(message_id, result=None, error=None) -> dict | None:
    if message_id is None:
        return None
    if error is not None:
        return {"jsonrpc": "2.0", "id": message_id, "error": error}
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def handle(message: dict) -> dict | None:
    method = message.get("method")
    message_id = message.get("id")
    params = message.get("params") or {}
    if method == "initialize":
        return _respond(
            message_id,
            {
                "protocolVersion": PROTOCOL,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "innovatech-hr", "version": "1.0.0"},
            },
        )
    if method in {"notifications/initialized", "initialized"}:
        return None
    if method == "ping":
        return _respond(message_id, {})
    if method == "tools/list":
        return _respond(message_id, {"tools": TOOL_SPECS})
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        payload = call_tool(name, arguments)
        is_error = isinstance(payload, dict) and payload.get("error") in {"unknown_tool", "invalid_arguments"}
        return _respond(
            message_id,
            {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "isError": is_error,
            },
        )
    if method and method.startswith("notifications/"):
        return None
    return _respond(message_id, error={"code": -32601, "message": f"Method not found: {method}"})


def main() -> None:
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            sys.stderr.write("Ignoring non-JSON MCP message\n")
            continue
        response = handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
