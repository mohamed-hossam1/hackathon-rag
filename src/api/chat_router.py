import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.auth_router import get_supabase_service, get_current_user
from src.api.auth_schemas import UserResponse
from src.db.supabase_service import SupabaseService
from src.services.title_service import ChatTitleService

logger = logging.getLogger("medical_rag.api.chat")
router = APIRouter(prefix="/chats", tags=["Chat Management & Persistence"])


class CreateChatRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="Optional chat title")


class ChatResponse(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    pinned: bool = False
    last_message_at: Optional[str]
    created_at: Optional[str]


class CreateMessageRequest(BaseModel):
    sender: str = Field(..., description="'user' or 'assistant'")
    content: str
    metadata: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    id: str
    chat_id: str
    sender: str
    content: str
    metadata: Optional[Dict[str, Any]]
    created_at: str


class GenerateTitleRequest(BaseModel):
    message: str = Field(..., description="First user message to derive the chat title from")


class GenerateTitleResponse(BaseModel):
    title: str


def get_title_service() -> ChatTitleService:
    return ChatTitleService()


@router.get("/", response_model=List[ChatResponse])
async def list_chats(
    current_user: UserResponse = Depends(get_current_user),
    supabase_service: SupabaseService = Depends(get_supabase_service)
) -> List[ChatResponse]:
    try:
        chats = supabase_service.list_user_chats(current_user.user_id)
        return chats
    except Exception as exc:
        logger.error(f"Failed to list chats for user {current_user.user_id}: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list chats")


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def create_chat(
    payload: CreateChatRequest,
    current_user: UserResponse = Depends(get_current_user),
    supabase_service: SupabaseService = Depends(get_supabase_service)
) -> ChatResponse:
    try:
        created = supabase_service.create_chat(user_id=current_user.user_id, title=payload.title)
        if not created:
            raise RuntimeError("Chat creation failed")
        return created
    except Exception as exc:
        logger.error(f"Failed to create chat for user {current_user.user_id}: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create chat")


@router.post("/{chat_id}/auto-title", response_model=GenerateTitleResponse, status_code=status.HTTP_200_OK)
async def auto_title_chat(
    chat_id: str,
    payload: GenerateTitleRequest,
    current_user: UserResponse = Depends(get_current_user),
    supabase_service: SupabaseService = Depends(get_supabase_service),
    title_service: ChatTitleService = Depends(get_title_service)
) -> GenerateTitleResponse:
    """Generates a concise title from the chat's first message and renames the chat."""
    try:
        title = title_service.generate_title(payload.message)
        ok = supabase_service.rename_chat(chat_id, title=title)
        if not ok:
            raise RuntimeError("Rename failed")
        return GenerateTitleResponse(title=title)
    except Exception as exc:
        logger.error(f"Failed to auto-title chat {chat_id}: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate chat title")


@router.get("/{chat_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    chat_id: str,
    supabase_service: SupabaseService = Depends(get_supabase_service)
) -> List[MessageResponse]:
    try:
        msgs = supabase_service.get_chat_messages(chat_id)
        return msgs
    except Exception as exc:
        logger.error(f"Failed to fetch messages for chat {chat_id}: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch messages")


@router.post("/{chat_id}/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def post_message(
    chat_id: str,
    payload: CreateMessageRequest,
    current_user: UserResponse = Depends(get_current_user),
    supabase_service: SupabaseService = Depends(get_supabase_service)
) -> MessageResponse:
    try:
        created = supabase_service.add_message(chat_id=chat_id, sender=payload.sender, content=payload.content, metadata=payload.metadata)
        if not created:
            raise RuntimeError("Failed to add message")
        return created
    except Exception as exc:
        logger.error(f"Failed to post message to chat {chat_id}: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to post message")


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: str,
    current_user: UserResponse = Depends(get_current_user),
    supabase_service: SupabaseService = Depends(get_supabase_service)
):
    try:
        ok = supabase_service.soft_delete_chat(chat_id)
        if not ok:
            raise RuntimeError("Delete failed")
        return None
    except Exception as exc:
        logger.error(f"Failed to delete chat {chat_id}: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete chat")


@router.patch("/{chat_id}/pin", status_code=status.HTTP_200_OK)
async def pin_chat(
    chat_id: str,
    pinned: bool = True,
    current_user: UserResponse = Depends(get_current_user),
    supabase_service: SupabaseService = Depends(get_supabase_service)
):
    try:
        ok = supabase_service.pin_chat(chat_id, pinned=pinned)
        if not ok:
            raise RuntimeError("Pin failed")
        return {"pinned": pinned}
    except Exception as exc:
        logger.error(f"Failed to pin chat {chat_id}: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to pin chat")


@router.patch("/{chat_id}/rename", status_code=status.HTTP_200_OK)
async def rename_chat(
    chat_id: str,
    title: str,
    current_user: UserResponse = Depends(get_current_user),
    supabase_service: SupabaseService = Depends(get_supabase_service)
):
    try:
        ok = supabase_service.rename_chat(chat_id, title=title)
        if not ok:
            raise RuntimeError("Rename failed")
        return {"title": title}
    except Exception as exc:
        logger.error(f"Failed to rename chat {chat_id}: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to rename chat")