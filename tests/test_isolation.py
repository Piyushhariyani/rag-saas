from types import SimpleNamespace

import pytest

TENANTS = ["tenant_a", "tenant_b", "tenant_c"]

# Topics belonging to *other* tenants, to tempt the retriever across namespaces.
QUERIES = [
    "What is the leave policy for employees?",
    "What are the terms of service and privacy rules?",
    "How do I clean the device and set it up?",
]


# ---- Retrieval isolation (uses embedding calls, no chat model) ----

@pytest.mark.parametrize("tenant", TENANTS)
@pytest.mark.parametrize("query", QUERIES)
def test_retriever_returns_only_own_chunks(tenant, query):
    # pyrefly: ignore [missing-import]
    from app.chain import get_retriever

    docs = get_retriever(tenant).invoke(query)
    for d in docs:
        assert d.metadata.get("tenant_id") == tenant, (
            f"{tenant} retrieved a chunk from {d.metadata.get('tenant_id')}"
        )


def test_unknown_tenant_gets_nothing():
    # pyrefly: ignore [missing-import]
    from app.chain import get_retriever

    assert get_retriever("tenant_does_not_exist").invoke("anything") == []


# ---- Feedback ownership (mocked LangSmith, no network) ----

class FakeLangSmith:
    def __init__(self, owner):
        self.owner = owner
        self.feedback = []

    def read_run(self, run_id):
        return SimpleNamespace(extra={"metadata": {"tenant_id": self.owner}})

    def create_feedback(self, **kwargs):
        self.feedback.append(kwargs)


def test_tenant_cannot_rate_another_tenants_run(client, monkeypatch, key_b):
    # pyrefly: ignore [missing-import]
    import app.server as server

    fake = FakeLangSmith(owner="tenant_a")
    monkeypatch.setattr(server, "ls_client", fake)
    monkeypatch.setattr(server.time, "sleep", lambda s: None)

    r = client.post(
        "/feedback",
        headers={"x-api-key": key_b},
        json={"run_id": "run-1", "score": 1},
    )
    assert r.status_code == 404
    assert fake.feedback == []          # nothing was written


def test_tenant_can_rate_its_own_run(client, monkeypatch, key_a):
    # pyrefly: ignore [missing-import]
    import app.server as server

    fake = FakeLangSmith(owner="tenant_a")
    monkeypatch.setattr(server, "ls_client", fake)

    r = client.post(
        "/feedback",
        headers={"x-api-key": key_a},
        json={"run_id": "run-1", "score": 0, "comment": "test"},
    )
    assert r.status_code == 200
    assert fake.feedback[0]["run_id"] == "run-1"


def test_feedback_score_is_validated(client, key_a):
    r = client.post(
        "/feedback",
        headers={"x-api-key": key_a},
        json={"run_id": "run-1", "score": 7},
    )
    assert r.status_code == 422


# ---- Ingest ties uploads to the key's tenant (mocked, no embeddings) ----

def test_ingest_uses_tenant_from_key_not_request(client, monkeypatch, key_b):
    # pyrefly: ignore [missing-import]
    import app.server as server

    calls = []

    def fake_ingest(path, tenant_id):
        calls.append((path.name, tenant_id))
        return 1

    monkeypatch.setattr(server, "ingest_file", fake_ingest)

    r = client.post(
        "/ingest?tenant_id=tenant_a",                 # attacker tries to choose
        headers={"x-api-key": key_b},
        files={"file": ("note.txt", b"hello")},
        data={"tenant_id": "tenant_a"},
    )
    assert r.status_code == 200
    assert r.json()["tenant"] == "tenant_b"
    assert calls == [("note.txt", "tenant_b")]


def test_ingest_rejects_unsupported_type(client, key_a):
    r = client.post(
        "/ingest",
        headers={"x-api-key": key_a},
        files={"file": ("virus.exe", b"MZ")},
    )
    assert r.status_code == 400


def test_ingest_strips_path_traversal(client, monkeypatch, key_a):
    # pyrefly: ignore [missing-import]
    import app.server as server

    seen = []
    monkeypatch.setattr(
        server, "ingest_file", lambda path, tenant_id: seen.append(path) or 1
    )

    r = client.post(
        "/ingest",
        headers={"x-api-key": key_a},
        files={"file": ("../../evil.txt", b"hello")},
    )
    assert r.status_code == 200
    assert seen[0].name == "evil.txt"       # directory parts were removed
    assert ".." not in str(seen[0])