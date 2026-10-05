import os
import tempfile
from pathlib import Path

import pytest

_RUNTIME = Path(tempfile.mkdtemp(prefix="innovatech-test-"))
os.environ["INDEX_PATH"] = str(_RUNTIME / "index.json")
os.environ["TICKETS_PATH"] = str(_RUNTIME / "tickets.json")
os.environ["LLM_DISABLED"] = "1"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
