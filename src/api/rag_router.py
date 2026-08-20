import logging
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.schemas import ErrorResponse, RAGRequest
from src.models.response import RAGResponse
from src.services.rag_service import RAGService

logger = logging.getLogger("medical_rag.api.rag")

router = APIRouter(tags=["RAG"])


# Simple dependency provider for RAGService instance
def get_rag_service() -> RAGService:
    return RAGService()


@router.post(
    "/rag",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request — empty query"},
        503: {"model": ErrorResponse, "description": "Service Unavailable — dependency failure"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def query_rag(
    request: RAGRequest,
    background_tasks: BackgroundTasks,
    rag_service: RAGService = Depends(get_rag_service)
) -> RAGResponse:
    """Executes medical RAG pipeline query and returns grounded answer with citations and confidence metrics."""
    if not request.query or not request.query.strip():
        logger.warning("Rejected empty RAG query request")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty"
        )

    try:
        response = rag_service.query(
            query_text=request.query,
            dev=request.dev,
            personal_context=request.personal_context,
            background_tasks=background_tasks
        )
        return response
    except RuntimeError as rerr:
        err_msg = str(rerr).lower()
        if "vector" in err_msg or "qdrant" in err_msg:
            logger.error(f"Vector database error during /rag query: {rerr}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector database temporarily unavailable"
            ) from rerr
        elif "llm" in err_msg or "rate limit" in err_msg or "connection" in err_msg:
            logger.error(f"LLM service error during /rag query: {rerr}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service temporarily unavailable"
            ) from rerr
        else:
            logger.error(f"Runtime error during /rag query: {rerr}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            ) from rerr
    except Exception as exc:
        logger.error(f"Unexpected exception during /rag query: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        ) from exc


@router.post(
    "/rag/stream",
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request — empty query"},
        503: {"model": ErrorResponse, "description": "Service Unavailable — dependency failure"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def query_rag_stream(
    request: RAGRequest,
    background_tasks: BackgroundTasks,
    rag_service: RAGService = Depends(get_rag_service)
) -> StreamingResponse:
    """Streams medical RAG pipeline answer token-by-token using Server-Sent Events (SSE)."""
    if not request.query or not request.query.strip():
        logger.warning("Rejected empty streaming RAG query request")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty"
        )

    try:
        generator = rag_service.query_stream(
            query_text=request.query,
            dev=request.dev,
            personal_context=request.personal_context,
            background_tasks=background_tasks
        )
        return StreamingResponse(generator, media_type="text/event-stream")
    except Exception as exc:
        logger.error(f"Unexpected exception during /rag/stream query: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        ) from exc

