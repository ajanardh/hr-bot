"""Paths and fixed seeds. Secrets come from the environment, never from source."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EMBED_DIM = 384


def index_path() -> Path:
    return Path(os.environ.get("INDEX_PATH", ROOT / "data" / "index.json"))


def tickets_path() -> Path:
    return Path(os.environ.get("TICKETS_PATH", ROOT / "mock_data" / "tickets.json"))


def employees_path() -> Path:
    return Path(os.environ.get("EMPLOYEES_PATH", ROOT / "mock_data" / "employees.json"))


def passwords_path() -> Path:
    return Path(os.environ.get("PASSWORDS_PATH", ROOT / "mock_data" / "passwords.json"))


def session_secret() -> str:
    return os.environ.get("SESSION_SECRET", "innovatech-dev-session")


def corpus_dirs() -> list[Path]:
    raw = os.environ.get("CORPUS_DIRS")
    if raw:
        return [Path(part) for part in raw.split(os.pathsep) if part]
    return [ROOT, ROOT / "corpus"]


def retrieval_k() -> int:
    return int(os.environ.get("RETRIEVAL_K", "4"))
