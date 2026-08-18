from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from src.models.citation import Citation, CitationValidationResult
from src.models.retrieval import RerankResult, RetrievalResult

DEFAULT_MEDICAL_DISCLAIMER = (
    "This response is generated for educational and informational purposes only based on ingested medical guidelines "
    "and does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider."
)


class ConfidenceLabel(str, Enum):
    """Human-readable evidence confidence classification."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class DevTrace(BaseModel):
    """Detailed retrieval, reranking, and citation validation trace returned in dev mode."""
    semantic_results: List[RetrievalResult] = Field(default_factory=list, description="Results from semantic retrieval")
    recursive_results: List[RetrievalResult] = Field(default_factory=list, description="Results from recursive retrieval")
    bm25_results: List[RetrievalResult] = Field(default_factory=list, description="Results from BM25 retrieval")
    deduplicated_candidates: List[RetrievalResult] = Field(default_factory=list, description="Candidates remaining after deduplication")
    reranker_results: List[RerankResult] = Field(default_factory=list, description="Results scored by cross-encoder reranker")
    selected_context: List[RerankResult] = Field(default_factory=list, description="Top-K context chunks selected for generation")
    citation_validations: List[CitationValidationResult] = Field(default_factory=list, description="All citation validation outcomes")


class RAGResponse(BaseModel):
    """Complete answer response returned by the RAG service."""
    answer: str = Field(..., description="Generated answer text or abstention message")
    citations: List[Citation] = Field(default_factory=list, description="List of citations for claims in the answer")
    citation_validations: Optional[List[CitationValidationResult]] = Field(default=None, description="Validation details for each citation")
    evidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall evidence support ratio (0.0 - 1.0)")
    confidence_label: ConfidenceLabel = Field(..., description="Confidence label based on evidence score")
    abstained: bool = Field(..., description="True if system abstained due to insufficient evidence")
    disclaimer: str = Field(default=DEFAULT_MEDICAL_DISCLAIMER, description="Mandatory medical safety disclaimer")
    dev_trace: Optional[DevTrace] = Field(default=None, description="Diagnostic step-by-step trace when dev=true")
