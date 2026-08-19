from uuid import uuid4
from fastapi.testclient import TestClient

from main import app
from src.auth import ApplicationRole, Principal, get_current_principal


def test_health_endpoint_public(test_client: TestClient):
    response = test_client.get("/health")
    assert response.status_code == 200


def test_rag_endpoint_unauthenticated_returns_401(test_client: TestClient):
    response = test_client.post("/rag", json={"query": "What is hypertension?"})
    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"


def test_rag_user_dev_true_returns_403(test_client: TestClient):
    user = Principal(user_id=uuid4(), role=ApplicationRole.USER)
    app.dependency_overrides[get_current_principal] = lambda: user
    try:
        response = test_client.post(
            "/rag",
            json={"query": "What is hypertension?", "dev": True},
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 403
        assert "Dev mode requires Admin privileges" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_current_principal, None)


def test_evaluate_endpoint_user_returns_403(test_client: TestClient):
    user = Principal(user_id=uuid4(), role=ApplicationRole.USER)
    app.dependency_overrides[get_current_principal] = lambda: user
    try:
        response = test_client.post(
            "/evaluate",
            json={"queries": ["test query"]},
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(get_current_principal, None)


def test_dev_queries_endpoint_user_returns_403(test_client: TestClient):
    user = Principal(user_id=uuid4(), role=ApplicationRole.USER)
    app.dependency_overrides[get_current_principal] = lambda: user
    try:
        response = test_client.get(
            "/dev/queries",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(get_current_principal, None)
