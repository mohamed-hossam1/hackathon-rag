from unittest.mock import MagicMock, patch
from uuid import UUID

import jwt
import pytest
from fastapi import HTTPException

from tests.conftest import (
    TEST_KID,
    TEST_PUBLIC_KEY,
    create_test_jwt,
)
from src.auth import ApplicationRole, Principal, verify_jwt_token
from src.config import AppConfig


@pytest.fixture
def mock_config():
    config = AppConfig()
    config.SUPABASE_URL = "https://nzaaikgnmqdwqaszdkgj.supabase.co"
    config.SUPABASE_JWT_AUDIENCE = "authenticated"
    config.SUPABASE_JWT_ALGORITHMS = ["ES256"]
    return config


@pytest.fixture(autouse=True)
def mock_jwk_client(mock_config):
    mock_key = MagicMock()
    mock_key.key = TEST_PUBLIC_KEY
    mock_client = MagicMock()
    mock_client.get_signing_key_from_jwt.return_value = mock_key

    with patch("src.auth.get_jwk_client", return_value=mock_client):
        yield mock_client


def test_verify_valid_user_token(mock_config):
    token = create_test_jwt(app_role="user")
    principal = verify_jwt_token(token, config=mock_config)

    assert isinstance(principal, Principal)
    assert principal.role == ApplicationRole.USER
    assert principal.email == "test@example.com"


def test_verify_valid_admin_token(mock_config):
    token = create_test_jwt(app_role="admin")
    principal = verify_jwt_token(token, config=mock_config)

    assert isinstance(principal, Principal)
    assert principal.role == ApplicationRole.ADMIN


def test_user_metadata_admin_ignored(mock_config):
    """Verifies that setting app_role=admin in user_metadata does NOT grant Admin privileges."""
    token = create_test_jwt(user_role="admin")
    principal = verify_jwt_token(token, config=mock_config)

    assert principal.role == ApplicationRole.USER


def test_expired_token_rejected(mock_config):
    token = create_test_jwt(exp_offset=-100)
    with pytest.raises(HTTPException) as exc_info:
        verify_jwt_token(token, config=mock_config)
    assert exc_info.value.status_code == 401


def test_invalid_audience_rejected(mock_config):
    token = create_test_jwt(aud="wrong_audience")
    with pytest.raises(HTTPException) as exc_info:
        verify_jwt_token(token, config=mock_config)
    assert exc_info.value.status_code == 401


def test_invalid_issuer_rejected(mock_config):
    token = create_test_jwt(iss="https://malicious-issuer.com")
    with pytest.raises(HTTPException) as exc_info:
        verify_jwt_token(token, config=mock_config)
    assert exc_info.value.status_code == 401


def test_unauthenticated_supabase_role_rejected(mock_config):
    token = create_test_jwt(role="anon")
    with pytest.raises(HTTPException) as exc_info:
        verify_jwt_token(token, config=mock_config)
    assert exc_info.value.status_code == 401
