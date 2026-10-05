"""Web chat, /chat, and /health for the Innovatech HR agent."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from app.agent import Agent
from app.auth import verify_password
from app.config import session_secret
from app.data_store import find_employee
from app.llm import synthesis_enabled
from app.mcp_client import McpClient, McpError
from app.rag.store import build_index, load_index

STATIC = Path(__file__).resolve().parent / "static"

DEMO_TASKS = [
    {
        "id": "pto-request",
        "title": "PTO request",
        "employee_id": "101",
        "message": "Can I take three days of PTO next week?",
        "expects": "Profile, PTO balance, Policy 200, a draft note to the manager, and no approval.",
    },
    {
        "id": "remote-six-weeks",
        "title": "Remote work for six weeks",
        "employee_id": "101",
        "message": "Can I work remotely from another U.S. state for six weeks?",
        "expects": "Profile, Policies 300 and 600, a change-of-location requirement, and a draft that is not sent.",
    },
]


class ChatRequest(BaseModel):
    message: str = ""
    employee_id: str | None = None
    confirm_action: bool = False
    pending_action: dict | None = None
    selected_step: dict | None = None


class LoginRequest(BaseModel):
    employee_id: str
    password: str


def _actor(request: Request) -> str | None:
    value = request.session.get("employee_id")
    return str(value) if value else None


def _profile(employee_id: str) -> dict | None:
    person = find_employee(employee_id=employee_id)
    if not isinstance(person, dict):
        return None
    return {
        "employee_id": person["employee_id"],
        "full_name": person["full_name"],
        "role": person["role"],
        "department": person["department"],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()
    client = McpClient()
    app.state.mcp = None
    app.state.agent = None
    app.state.mcp_error = None
    try:
        client.start()
        app.state.mcp = client
        app.state.agent = Agent(client)
    except McpError as exc:
        app.state.mcp_error = str(exc)
        client.close()
    yield
    if app.state.mcp:
        app.state.mcp.close()


app = FastAPI(title="Innovatech HR Agent", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=session_secret(), same_site="lax", https_only=False)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health(request: Request) -> dict:
    mcp: McpClient | None = request.app.state.mcp
    connected = bool(mcp and mcp.healthy())
    index = load_index()
    return {
        "status": "ok" if connected else "degraded",
        "mcp": "connected" if connected else "unavailable",
        "mcp_tools": len(mcp.tools) if mcp else 0,
        "index_chunks": index.get("chunk_count", 0),
        "llm_synthesis": synthesis_enabled(),
        "detail": request.app.state.mcp_error,
    }


@app.post("/login")
def login(body: LoginRequest, request: Request):
    employee_id = body.employee_id.strip()
    if not verify_password(employee_id, body.password):
        return JSONResponse({"detail": "Employee id or password is incorrect."}, status_code=401)
    profile = _profile(employee_id)
    if profile is None:
        return JSONResponse({"detail": "Employee id or password is incorrect."}, status_code=401)
    request.session["employee_id"] = profile["employee_id"]
    return profile


@app.post("/logout")
def logout(request: Request) -> dict:
    request.session.clear()
    return {"status": "signed_out"}


@app.get("/api/me")
def me(request: Request):
    profile = _profile(_actor(request) or "")
    if profile is None:
        return JSONResponse({"detail": "Sign in required."}, status_code=401)
    return profile


@app.get("/api/demo-tasks")
def demo_tasks() -> dict:
    return {"tasks": DEMO_TASKS}


@app.post("/chat")
def chat(body: ChatRequest, request: Request):
    agent: Agent | None = request.app.state.agent
    if agent is None:
        return JSONResponse(
            {
                "answer": "The policy tools are unavailable because the MCP server did not start. Retry in a moment. Nothing was changed.",
                "intent": "error",
                "citations": [],
                "snippets": [],
                "trace": [
                    {
                        "step": 1,
                        "tool": None,
                        "arguments": {},
                        "status": "error",
                        "output": {"error": "mcp_unavailable", "detail": request.app.state.mcp_error},
                    }
                ],
                "escalation": None,
                "pending_action": None,
                "action_taken": False,
                "llm_synthesis": False,
            }
        )
    actor = _actor(request)
    if not actor:
        return JSONResponse({"detail": "Sign in required."}, status_code=401)
    return agent.run(
        message=body.message,
        employee_id=actor,
        confirm_action=body.confirm_action,
        pending_action=body.pending_action,
        actor_employee_id=actor,
        selected_step=body.selected_step,
    )
