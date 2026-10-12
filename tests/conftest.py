from dotenv import load_dotenv
load_dotenv(override=True)

import os
import pytest
from fastapi.testclient import TestClient


def _key(tenant: str) -> str:
    key = os.getenv(f"KEY_{tenant.upper()}")
    if not key:
        pytest.skip(f"KEY_{tenant.upper()} not set in .env")
    return key


@pytest.fixture(scope="session")
def client():
    from app.server import app
    return TestClient(app)


@pytest.fixture(scope="session")
def key_a():
    return _key("tenant_a")


@pytest.fixture(scope="session")
def key_b():
    return _key("tenant_b")


@pytest.fixture(scope="session")
def key_c():
    return _key("tenant_c")