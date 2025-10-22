from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_evaluate_allows_clean_text():
    r = client.post("/evaluate", json={"text": "hello world", "policy": None})
    assert r.status_code == 200
    body = r.json()
    assert body["allowed"] is True
    assert "No issues" in body["reason"]

def test_evaluate_blocks_policy_match():
    r = client.post("/evaluate", json={"text": "this has a forbidden token", "policy": "forbidden"})
    assert r.status_code == 200
    body = r.json()
    assert body["allowed"] is False
    assert "Blocked by policy" in body["reason"] or "forbidden keyword" in body["reason"]
