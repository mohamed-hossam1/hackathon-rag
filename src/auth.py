import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from src.config import AppConfig, get_config

logger = logging.getLogger("medical_rag.auth")

security_scheme = HTTPBearer(auto_error=False)


class ApplicationRole(str, Enum):
    USER = "user"
    ADMIN = "admin"


@dataclass(frozen=True)
class Principal:
    user_id: UUID
    email: Optional[str] = None
    role: ApplicationRole = ApplicationRole.USER


_jwk_client: Optional[PyJWKClient] = None


def get_jwk_client(config: AppConfig) -> PyJWKClient:
    """Returns a cached PyJWKClient instance for the configured Supabase JWKS endpoint."""
    global _jwk_client
    if _jwk_client is None:
        _jwk_client = PyJWKClient(
            config.SUPABASE_JWKS_URL,
            cache_keys=True,
            max_cached_keys=10,
        )
    return _jwk_client


def verify_jwt_token(token: str, config: Optional[AppConfig] = None) -> Principal:
    """
    Verifies an asymmetric JWT token issued by Supabase Auth using JWKS (ES256).
    Derives a trusted Principal with an application role mapped strictly from app_metadata.app_role.
    """
    if config is None:
        config = get_config()

    try:
        jwk_client = get_jwk_client(config)
        signing_key = jwk_client.get_signing_key_from_jwt(token)

        payload: dict[str, Any] = jwt.decode(
            token,
            signing_key.key,
            algorithms=config.SUPABASE_JWT_ALGORITHMS,
            audience=config.SUPABASE_JWT_AUDIENCE,
            issuer=config.SUPABASE_JWT_ISSUER,
            leeway=config.JWT_CLOCK_SKEW_SECONDS,
            options={
                "require": ["exp", "iss", "aud", "sub"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_iss": True,
                "verify_aud": True,
            },
        )
    except jwt.PyJWTError as exc:
        logger.warning(f"JWT validation failed: {exc.__class__.__name__}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as exc:
        logger.error(f"Unexpected error during JWT verification: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 1. Require Supabase DB role == "authenticated"
    supabase_role = payload.get("role")
    if supabase_role != "authenticated":
        logger.warning("JWT rejected: invalid Supabase database role claim")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2. Extract and validate user_id (sub claim) as a UUID
    sub_str = payload.get("sub")
    if not sub_str:
        logger.warning("JWT rejected: missing subject claim")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = UUID(sub_str)
    except (ValueError, TypeError):
        logger.warning("JWT rejected: malformed subject UUID")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    email = payload.get("email")

    # 3. Derive application role strictly from trusted app_metadata.app_role
    app_metadata = payload.get("app_metadata", {})
    if isinstance(app_metadata, dict) and app_metadata.get("app_role") == "admin":
        role = ApplicationRole.ADMIN
    else:
        role = ApplicationRole.USER

    return Principal(user_id=user_id, email=email, role=role)


async def get_current_principal(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    config: AppConfig = Depends(get_config),
) -> Principal:
    """FastAPI dependency to enforce authentication and extract the verified Principal."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return verify_jwt_token(credentials.credentials, config=config)


async def require_admin(
    principal: Principal = Depends(get_current_principal),
) -> Principal:
    """FastAPI dependency to enforce Admin role authorization."""
    if principal.role != ApplicationRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return principal


def enforce_rag_dev_policy(dev_enabled: bool, principal: Principal) -> None:
    """Enforces that dev=True mode is strictly restricted to Admins before processing RAG requests."""
    if dev_enabled and principal.role != ApplicationRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Dev mode requires Admin privileges.",
        )
