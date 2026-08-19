import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from pydantic import BaseModel, Field

from src.api.schemas import ErrorResponse
from src.auth import Principal, require_admin
from src.db.supabase_service import SupabaseService
from src.evaluation.llm_judge import LLMJudge
from src.models.chunk import Chunk, ChunkerType

logger = logging.getLogger("medical_rag.api.dev_router")

router = APIRouter(prefix="/dev", tags=["Dev Dashboard & Trace Logs"])

supabase_service = SupabaseService()
llm_judge = LLMJudge()


# Response Models
class QuerySummaryResponse(BaseModel):
    id: str
    query_text: str
    created_at: Optional[str] = None
    abstained: bool = False
    evidence_score: float = 0.0


class ChunkTraceItem(BaseModel):
    rank: int
    chunk_id: str
    chunk_text: str
    score: float = 0.0
    filename: Optional[str] = None
    page_start: Optional[int] = None


class TraceDetailResponse(BaseModel):
    query_id: str
    query_text: str
    created_at: Optional[str] = None
    abstained: bool = False
    evidence_score: float = 0.0
    traces: Dict[str, List[ChunkTraceItem]]


class ChunkJudgmentItem(BaseModel):
    chunk_id: str
    relevant: bool
    reason: str


class MethodEvaluationResult(BaseModel):
    retrieval_method: str
    precision_at_3: float  # e.g., 66.67 (formatted as percentage or fraction 0.67)
    precision_at_5: float
    judgments: List[ChunkJudgmentItem]


class QueryEvaluationReport(BaseModel):
    query_id: str
    query_text: str
    methods: List[MethodEvaluationResult]


@router.get(
    "/queries",
    response_model=List[QuerySummaryResponse],
    summary="List Logged Dev Queries",
    description="Returns a list of all query runs logged in Supabase during dev mode."
)
async def list_logged_queries(
    limit: int = Query(default=50, ge=1, le=200),
    admin: Principal = Depends(require_admin),
):
    if not supabase_service.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Supabase credentials (SUPABASE_URL, SUPABASE_KEY) are not configured."
        )

    queries = supabase_service.get_queries(limit=limit)
    return queries


@router.get(
    "/queries/{query_id}/trace",
    response_model=TraceDetailResponse,
    summary="Get Full Retrieval Trace for a Query",
    description="Returns stored top 10 chunks per retrieval strategy (semantic, recursive, bm25, reranker) with chunk_id and text content."
)
async def get_query_trace(
    query_id: str = Path(..., description="UUID of logged dev query"),
    admin: Principal = Depends(require_admin),
):
    if not supabase_service.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Supabase credentials are not configured."
        )

    detail = supabase_service.get_query_traces(query_id)
    if not detail or not detail.get("query_text"):
        raise HTTPException(
            status_code=404,
            detail=f"Query trace for query_id '{query_id}' not found."
        )

    return detail


@router.post(
    "/queries/{query_id}/evaluate",
    response_model=QueryEvaluationReport,
    summary="Run AI-as-a-Judge Evaluation across Strategies",
    description="Evaluates top 5 chunks per strategy using exactly 4 batch LLM calls. Computes Precision@3 and Precision@5 and saves to Supabase."
)
async def evaluate_query_retrieval(
    query_id: str = Path(..., description="UUID of logged dev query"),
    admin: Principal = Depends(require_admin),
):

    if not supabase_service.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Supabase credentials are not configured."
        )

    # 1. Fetch trace data for query
    detail = supabase_service.get_query_traces(query_id)
    if not detail or not detail.get("query_text"):
        raise HTTPException(
            status_code=404,
            detail=f"Query trace for query_id '{query_id}' not found."
        )

    query_text = detail.get("query_text", "")
    traces = detail.get("traces", {})

    strategies = ["semantic", "recursive", "bm25", "reranker"]
    method_eval_results: List[MethodEvaluationResult] = []

    # Check if reports already exist in Supabase
    existing_reports = supabase_service.get_evaluation_reports(query_id)
    existing_by_method = {r.get("retrieval_method"): r for r in existing_reports}

    for method in strategies:
        # If already evaluated in Supabase, return cached report
        if method in existing_by_method:
            r = existing_by_method[method]
            judgments = [
                ChunkJudgmentItem(
                    chunk_id=j.get("chunk_id", ""),
                    relevant=j.get("relevant", False),
                    reason=j.get("reason", "")
                )
                for j in r.get("judgments", [])
            ]
            method_eval_results.append(
                MethodEvaluationResult(
                    retrieval_method=method,
                    precision_at_3=round(float(r.get("precision_at_3", 0.0)), 2),
                    precision_at_5=round(float(r.get("precision_at_5", 0.0)), 2),
                    judgments=judgments
                )
            )
            continue

        # Get top 5 stored trace items for this method
        items = traces.get(method, [])[:5]
        if not items:
            method_eval_results.append(
                MethodEvaluationResult(
                    retrieval_method=method,
                    precision_at_3=0.0,
                    precision_at_5=0.0,
                    judgments=[]
                )
            )
            continue

        # Convert trace items to Chunk objects for LLMJudge
        chunks = [
            Chunk(
                chunk_id=item.get("chunk_id", f"{method}_{idx}"),
                text=item.get("chunk_text") or item.get("text") or "N/A",
                document_id=item.get("document_id", "doc_unknown"),
                filename=item.get("filename", "doc.pdf"),
                page_start=item.get("page_start", 1),
                page_end=item.get("page_start", 1),
                chunk_index=item.get("chunk_index", idx),
                chunker_type=item.get("chunker_type", ChunkerType.SEMANTIC if method == "semantic" else ChunkerType.RECURSIVE),
                start_char=item.get("start_char", 0),
                end_char=item.get("end_char", max(1, len(item.get("chunk_text") or item.get("text") or "N/A"))),
            )
            for idx, item in enumerate(items, 1)
        ]

        # Single batch LLM call for all 5 chunks of this method (1 call per method = 4 calls total)
        judgments_raw = llm_judge.judge_batch_by_method(
            query=query_text,
            chunks=chunks,
            retrieval_method=method
        )

        # Compute Precision@3 and Precision@5
        # Top 3 items
        rel_3 = sum(1 for j in judgments_raw[:3] if j.relevant)
        k_3 = min(len(judgments_raw), 3)
        p_at_3 = (rel_3 / k_3 * 100.0) if k_3 > 0 else 0.0

        # Top 5 items
        rel_5 = sum(1 for j in judgments_raw[:5] if j.relevant)
        k_5 = min(len(judgments_raw), 5)
        p_at_5 = (rel_5 / k_5 * 100.0) if k_5 > 0 else 0.0

        judgments_json = [
            {
                "chunk_id": j.chunk_id,
                "relevant": j.relevant,
                "reason": j.reason
            }
            for j in judgments_raw
        ]

        # Save report to Supabase
        supabase_service.save_evaluation_report(
            query_id=query_id,
            retrieval_method=method,
            precision_at_3=p_at_3,
            precision_at_5=p_at_5,
            judgments=judgments_json
        )

        judgments_items = [
            ChunkJudgmentItem(
                chunk_id=j.chunk_id,
                relevant=j.relevant,
                reason=j.reason
            )
            for j in judgments_raw
        ]

        method_eval_results.append(
            MethodEvaluationResult(
                retrieval_method=method,
                precision_at_3=round(p_at_3, 2),
                precision_at_5=round(p_at_5, 2),
                judgments=judgments_items
            )
        )

    return QueryEvaluationReport(
        query_id=query_id,
        query_text=query_text,
        methods=method_eval_results
    )
