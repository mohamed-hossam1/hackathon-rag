from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from src.models.citation import CitationValidationResult
from src.models.document import DocumentStatus
from src.models.evaluation import EvaluationReport
from src.models.retrieval import RerankResult, RetrievalResult


class RAGRequest(BaseModel):
    """Request payload for POST /rag endpoint."""
    query: str = Field(..., min_length=1, description="Medical query or question")
    dev: bool = Field(default=False, description="Enable dev mode to return step-by-step DevTrace")


class UploadResponse(BaseModel):
    """Response payload for POST /upload endpoint."""
    document_id: str = Field(..., description="UUID assigned to uploaded document")
    filename: str = Field(..., description="Original uploaded filename")
    file_type: str = Field(..., description="Detected file extension/type")
    status: DocumentStatus = Field(..., description="Current status (typically queued)")
    message: str = Field(..., description="Informational upload status message")


class DocumentStatusResponse(BaseModel):
    """Response payload for GET /documents/{document_id}/status endpoint."""
    document_id: str = Field(..., description="Target document UUID")
    filename: str = Field(..., description="Document filename")
    status: DocumentStatus = Field(..., description="Current document processing status")
    total_pages: Optional[int] = Field(default=None, description="Total pages extracted if completed")
    error_message: Optional[str] = Field(default=None, description="Failure reason if status is failed")


class DocumentItem(BaseModel):
    """Single document item representation for listing."""
    document_id: str = Field(..., description="Document UUID")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File format type")
    status: DocumentStatus = Field(..., description="Current status")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")
    total_pages: Optional[int] = Field(default=None, description="Total extracted pages")


class DocumentListResponse(BaseModel):
    """Response payload for GET /documents endpoint."""
    documents: List[DocumentItem] = Field(default_factory=list, description="List of uploaded documents")


class RetrieveRequest(BaseModel):
    """Request payload for debug retrieval endpoints."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=10, ge=1, description="Maximum results to retrieve")


class RetrieveResponse(BaseModel):
    """Response payload for debug retrieval endpoints."""
    results: List[RetrievalResult] = Field(default_factory=list, description="List of retrieval results")


class RerankRequest(BaseModel):
    """Request payload for POST /rerank endpoint."""
    query: str = Field(..., min_length=1, description="Search query")
    chunk_ids: List[str] = Field(..., description="List of candidate chunk IDs to rerank")
    top_k: int = Field(default=5, ge=1, description="Maximum top reranked results to return")


class RerankResponse(BaseModel):
    """Response payload for POST /rerank endpoint."""
    results: List[RerankResult] = Field(default_factory=list, description="List of reranked results")


class ClaimCitationPair(BaseModel):
    """Single claim and chunk ID pair for citation validation."""
    claim: str = Field(..., description="Claim text from generated answer")
    chunk_id: str = Field(..., description="Target cited chunk ID")


class ValidateCitationsRequest(BaseModel):
    """Request payload for POST /validate-citations endpoint."""
    validations: List[ClaimCitationPair] = Field(..., description="List of claim-citation pairs to validate")


class ValidateCitationsResponse(BaseModel):
    """Response payload for POST /validate-citations endpoint."""
    results: List[CitationValidationResult] = Field(default_factory=list, description="Validation results per citation")


class EvaluateRequest(BaseModel):
    """Request payload for POST /evaluate endpoint."""
    queries: List[str] = Field(..., min_items=1, description="List of evaluation queries to run")
    top_k: int = Field(default=5, ge=1, description="Top-K context size for evaluation")


class EvaluateResponse(BaseModel):
    """Response payload for POST /evaluate endpoint."""
    results: List[EvaluationReport] = Field(default_factory=list, description="Aggregated evaluation reports per query")


class HealthResponse(BaseModel):
    """Response payload for GET /health endpoint."""
    status: str = Field(default="healthy", description="Overall service status")
    qdrant_connected: bool = Field(..., description="Whether vector database connection is healthy")
    models_loaded: bool = Field(..., description="Whether embedding and reranker models are loaded")


class ErrorResponse(BaseModel):
    """Standard error response payload matching API contract specification."""
    detail: str = Field(..., description="Human-readable error explanation")

