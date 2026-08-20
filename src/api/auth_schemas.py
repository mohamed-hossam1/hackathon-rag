from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field


class SignUpRequest(BaseModel):
    """Request payload for POST /auth/signup endpoint."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="User password (at least 6 characters)")
    full_name: Optional[str] = Field(default=None, description="User full name")


class LoginRequest(BaseModel):
    """Request payload for POST /auth/login endpoint."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class UserResponse(BaseModel):
    """User profile response representation."""
    user_id: str = Field(..., description="Unique Supabase User UUID")
    email: str = Field(..., description="User email address")
    full_name: Optional[str] = Field(default=None, description="User full name if available")
    created_at: Optional[str] = Field(default=None, description="User registration timestamp")


class AuthResponse(BaseModel):
    """Authentication token response payload."""
    access_token: str = Field(..., description="Supabase JWT access token")
    refresh_token: Optional[str] = Field(default=None, description="Supabase refresh token")
    token_type: str = Field(default="bearer", description="Token type, typically 'bearer'")
    user: UserResponse = Field(..., description="Authenticated user details")


class SaveMemoryRequest(BaseModel):
    """Request payload for saving personal medical memory."""
    memory_text: str = Field(..., min_length=1, description="Personal medical context/memory text to save")


class MemoryItem(BaseModel):
    """Representation of a saved personal memory item."""
    id: str = Field(..., description="Memory record UUID")
    user_id: str = Field(..., description="Owner user UUID")
    memory_text: str = Field(..., description="Saved personal context memory text")
    created_at: str = Field(..., description="Creation timestamp")


class MemoryListResponse(BaseModel):
    """Response payload for fetching saved user memories."""
    memories: List[MemoryItem] = Field(default_factory=list, description="List of saved memory items")
    concatenated_context: str = Field(default="", description="Combined personal medical context string ready for RAG")
