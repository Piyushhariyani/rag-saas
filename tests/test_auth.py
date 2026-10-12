import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

# pyrefly: ignore [missing-import]
from app.auth import tenant_from_key


def _request_with_key(key: str | None) -> Request:
    headers = [(b"x-api-key", key.encode())] if key else []
    return Request({"type": "http", "method": "POST", "headers": headers})


# ---- key to tenant mapping (no network) ----

def test_missing_key_rejected():
    with pytest.raises(HTTPException) as e:
        tenant_from_key(None)
    assert e.value.status_code == 401


def test_wrong_key_rejected():
    with pytest.raises(HTTPException) as e:
        tenant_from_key("definitely-not-a-real-key")
    assert e.value.status_code == 401


def test_keys_map_to_their_own_tenant(key_a, key_b, key_c):
    assert tenant_from_key(key_a) == "tenant_a"
    assert tenant_from_key(key_b) == "tenant_b"
    assert tenant_from_key(key_c) == "tenant_c"


# ---- HTTP level ----

def test_health_is_open(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ingest_requires_key(client):
    r = client.post("/ingest", files={"file": ("a.txt", b"hello")})
    assert r.status_code == 401


def test_feedback_requires_key(client):
    r = client.post("/feedback", json={"run_id": "x", "score": 1})
    assert r.status_code == 401


def test_query_requires_key(client):
    r = client.post("/query/invoke", json={"input": {"question": "hi"}})
    assert r.status_code == 401


# ---- the important one: client cannot choose the tenant ----

def test_server_overwrites_client_tenant_id(key_a):
    # pyrefly: ignore [missing-import]
    from app.server import per_req_config_modifier

    config = {"configurable": {"tenant_id": "tenant_b"}}  # attacker's value
    result = asyncio.run(per_req_config_modifier(config, _request_with_key(key_a)))

    assert result["configurable"]["tenant_id"] == "tenant_a"
    assert result["metadata"]["tenant_id"] == "tenant_a"
    assert "tenant:tenant_a" in result["tags"]