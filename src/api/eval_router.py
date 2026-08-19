import logging
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas import ErrorResponse, EvaluateRequest, EvaluateResponse
from src.evaluation.rag_evaluator import RAGEvaluator

logger = logging.getLogger("medical_rag.api.eval")

router = APIRouter(tags=["Evaluation"])


def get_rag_evaluator() -> RAGEvaluator:
    return RAGEvaluator()


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request — empty query list"},
        503: {"model": ErrorResponse, "description": "Service Unavailable — dependency failure"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def evaluate_rag(
    request: EvaluateRequest,
    evaluator: RAGEvaluator = Depends(get_rag_evaluator)
) -> EvaluateResponse:
    """Evaluates retrieval quality across semantic, recursive, BM25, and reranker stages using LLM-as-a-Judge."""
    if not request.queries:
        logger.warning("Rejected empty evaluation query list")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Queries list must not be empty"
        )

    for q in request.queries:
        if not q.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Queries must contain non-empty strings"
            )

    try:
        reports = evaluator.evaluate_queries(queries=request.queries, top_k=request.top_k)
        return EvaluateResponse(results=reports)
    except RuntimeError as rerr:
        err_msg = str(rerr).lower()
        if "vector" in err_msg or "qdrant" in err_msg:
            logger.error(f"Vector database error during /evaluate: {rerr}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector database temporarily unavailable"
            ) from rerr
        elif "llm" in err_msg or "rate limit" in err_msg or "connection" in err_msg:
            logger.error(f"LLM service error during /evaluate: {rerr}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service temporarily unavailable"
            ) from rerr
        else:
            logger.error(f"Runtime error during /evaluate: {rerr}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            ) from rerr
    except Exception as exc:
        logger.error(f"Unexpected exception during /evaluate: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        ) from exc
