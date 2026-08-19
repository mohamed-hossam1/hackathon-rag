from typing import Any, Dict
from uuid import uuid4

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from src.auth import ApplicationRole
from src.config import AppConfig, get_config

# Generate ES256 key pair for test signing
TEST_PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())
TEST_PUBLIC_KEY = TEST_PRIVATE_KEY.public_key()
TEST_KID = "test-key-id-123"

TEST_ISSUER = "https://nzaaikgnmqdwqaszdkgj.supabase.co/auth/v1"
TEST_AUDIENCE = "authenticated"


def create_test_jwt(
    user_id: str | None = None,
    email: str | None = "test@example.com",
    role: str = "authenticated",
    app_role: str | None = None,
    user_role: str | None = None,
    iss: str = TEST_ISSUER,
    aud: str = TEST_AUDIENCE,
    exp_offset: int = 3600,
    alg: str = "ES256",
    kid: str = TEST_KID,
) -> str:
    """Helper to generate signed JWT tokens for unit/integration testing."""
    import time
    now = int(time.time())
    payload: Dict[str, Any] = {
        "sub": user_id or str(uuid4()),
        "email": email,
        "role": role,
        "iss": iss,
        "aud": aud,
        "iat": now,
        "exp": now + exp_offset,
        "app_metadata": {},
        "user_metadata": {},
    }

    if app_role is not None:
        payload["app_metadata"]["app_role"] = app_role

    if user_role is not None:
        payload["user_metadata"]["app_role"] = user_role

    headers = {"kid": kid, "alg": alg}
    return jwt.encode(payload, TEST_PRIVATE_KEY, algorithm=alg, headers=headers)


@pytest.fixture
def test_client():
    from main import app
    return TestClient(app)


@pytest.fixture
def valid_user_token():
    return create_test_jwt(app_role="user")


@pytest.fixture
def valid_admin_token():
    return create_test_jwt(app_role="admin")
