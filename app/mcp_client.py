"""Stdio MCP client. Tool calls go through this process boundary, not direct imports."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from app.config import ROOT


class McpError(RuntimeError):
    pass


class McpClient:
    def __init__(self, command: list[str] | None = None) -> None:
        self.command = command or [sys.executable, "-m", "mcp.server"]
        self.proc: subprocess.Popen | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._id = 0
        self._lock = threading.Lock()
        self.tools: list[dict] = []
        self.stderr_lines: list[str] = []

    def start(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        self.proc = subprocess.Popen(
            self.command,
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "innovatech-hr-agent", "version": "1.0.0"},
            },
        )
        self._notify("notifications/initialized", {})
        listed = self._request("tools/list", {})
        self.tools = listed.get("tools", [])

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self._queue.put(line)
        self._queue.put(None)

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip())
            if len(self.stderr_lines) > 50:
                self.stderr_lines = self.stderr_lines[-50:]

    def _send(self, payload: dict) -> None:
        if not self.proc or not self.proc.stdin:
            raise McpError("MCP server is not running")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict, timeout: float = 30) -> dict:
        with self._lock:
            self._id += 1
            message_id = self._id
            self._send({"jsonrpc": "2.0", "id": message_id, "method": method, "params": params})
            while True:
                try:
                    line = self._queue.get(timeout=timeout)
                except queue.Empty as exc:
                    detail = " ".join(self.stderr_lines[-5:])
                    raise McpError(f"Timed out waiting for {method}. {detail}") from exc
                if line is None:
                    detail = " ".join(self.stderr_lines[-5:])
                    raise McpError(f"MCP server exited during {method}. {detail}")
                data = json.loads(line)
                if data.get("id") != message_id:
                    continue
                if "error" in data:
                    raise McpError(str(data["error"]))
                return data.get("result") or {}

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        result = self._request("tools/call", {"name": name, "arguments": arguments or {}})
        content = result.get("content") or []
        texts = [item.get("text", "") for item in content if item.get("type") == "text"]
        raw = texts[0] if texts else "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
        if result.get("isError"):
            payload.setdefault("error", "tool_error")
        return payload

    def healthy(self) -> bool:
        if not self.proc or self.proc.poll() is not None:
            return False
        try:
            self._request("ping", {}, timeout=5)
            return True
        except McpError:
            return False

    def close(self) -> None:
        if not self.proc:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
