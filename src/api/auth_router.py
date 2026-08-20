import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Header, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.auth_schemas import (
    AuthResponse,
    LoginRequest,
    MemoryItem,
    MemoryListResponse,
    SaveMemoryRequest,
    SignUpRequest,
    UserResponse,
)
from src.db.supabase_service import SupabaseService

logger = logging.getLogger("medical_rag.api.auth")

router = APIRouter(prefix="/auth", tags=["Authentication & Personal Memory"])
security = HTTPBearer(auto_error=False)


def get_supabase_service() -> SupabaseService:
    return SupabaseService()


def parse_user_dict(raw_user: dict) -> UserResponse:
    """Helper to convert raw Supabase Auth user payload into UserResponse model."""
    user_id = raw_user.get("id", "")
    email = raw_user.get("email", "")
    metadata = raw_user.get("user_metadata", {}) or {}
    full_name = metadata.get("full_name") or metadata.get("name")
    created_at = raw_user.get("created_at")

    return UserResponse(
        user_id=user_id,
        email=email,
        full_name=full_name,
        created_at=created_at
    )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    supabase_service: SupabaseService = Depends(get_supabase_service)
) -> UserResponse:
    """FastAPI dependency enforcing valid Bearer JWT token authentication via Supabase."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = credentials.credentials
    try:
        raw_user = supabase_service.get_user_from_token(token)
        return parse_user_dict(raw_user)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(err),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as exc:
        logger.error(f"Failed to authenticate token: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    supabase_service: SupabaseService = Depends(get_supabase_service)
) -> Optional[UserResponse]:
    """FastAPI dependency for optional Bearer token authentication."""
    if not credentials or not credentials.credentials:
        return None

    token = credentials.credentials
    try:
        raw_user = supabase_service.get_user_from_token(token)
        return parse_user_dict(raw_user)
    except Exception:
        return None


@router.post(
    "/signup",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user account"
)
async def signup(
    payload: SignUpRequest,
    supabase_service: SupabaseService = Depends(get_supabase_service)
) -> AuthResponse:
    """Registers a new user account with Supabase Auth."""
    try:
        data = supabase_service.signup_user(
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name
        )

        user_data = data.get("user") or data
        user_response = parse_user_dict(user_data)
        access_token = data.get("access_token") or ""
        refresh_token = data.get("refresh_token") or None

        return AuthResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            user=user_response
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as exc:
        logger.error(f"Signup error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(exc)}"
        )


@router.post(
    "/login",
    response_model=AuthResponse,
    status_code=status.HTTP_200_OK,
    summary="User login & token generation"
)
async def login(
    payload: LoginRequest,
    supabase_service: SupabaseService = Depends(get_supabase_service)
) -> AuthResponse:
    """Authenticates user with email & password via Supabase Auth."""
    try:
        data = supabase_service.login_user(
            email=payload.email,
            password=payload.password
        )

        user_data = data.get("user") or {}
        user_response = parse_user_dict(user_data)
        access_token = data.get("access_token", "")
        refresh_token = data.get("refresh_token")

        return AuthResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            user=user_response
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(val_err)
        )
    except Exception as exc:
        logger.error(f"Login error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(exc)}"
        )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current user profile"
)
async def get_me(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """Returns profile details of the currently authenticated user."""
    return current_user


@router.post(
    "/memory",
    response_model=MemoryItem,
    status_code=status.HTTP_201_CREATED,
    summary="Save personal medical context memory"
)
async def save_memory(
    payload: SaveMemoryRequest,
    current_user: UserResponse = Depends(get_current_user),
    supabase_service: SupabaseService = Depends(get_supabase_service)
) -> MemoryItem:
    """Saves a personal medical background detail/memory for future chat sessions."""
    try:
        record = supabase_service.save_user_memory(
            user_id=current_user.user_id,
            memory_text=payload.memory_text
        )

        return MemoryItem(
            id=str(record.get("id", "")),
            user_id=str(record.get("user_id", current_user.user_id)),
            memory_text=str(record.get("memory_text", payload.memory_text)),
            created_at=str(record.get("created_at", ""))
        )
    except Exception as exc:
        logger.error(f"Error saving user memory: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save personal memory: {str(exc)}"
        )


@router.get(
    "/memory",
    response_model=MemoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch saved user medical memories"
)
async def get_memories(
    current_user: UserResponse = Depends(get_current_user),
    supabase_service: SupabaseService = Depends(get_supabase_service)
) -> MemoryListResponse:
    """Retrieves all saved personal medical memories for the authenticated user."""
    try:
        raw_items = supabase_service.get_user_memories(current_user.user_id)
        memories: List[MemoryItem] = []
        context_texts: List[str] = []

        for row in raw_items:
            m_text = row.get("memory_text", "")
            item = MemoryItem(
                id=str(row.get("id", "")),
                user_id=str(row.get("user_id", current_user.user_id)),
                memory_text=m_text,
                created_at=str(row.get("created_at", ""))
            )
            memories.append(item)
            if m_text.strip():
                context_texts.append(f"- {m_text.strip()}")

        concatenated_context = "\n".join(context_texts)

        return MemoryListResponse(
            memories=memories,
            concatenated_context=concatenated_context
        )
    except Exception as exc:
        logger.error(f"Error fetching user memories: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch memories: {str(exc)}"
        )


@router.delete(
    "/memory/{memory_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a saved personal memory item"
)
async def delete_memory(
    memory_id: str,
    current_user: UserResponse = Depends(get_current_user),
    supabase_service: SupabaseService = Depends(get_supabase_service)
):
    """Deletes a specific saved personal memory item."""
    success = supabase_service.delete_user_memory(current_user.user_id, memory_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete memory item"
        )
    return {"message": "Memory deleted successfully", "id": memory_id}
