from typing import Dict, List
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """The outcome of judging a retrieved chunk's relevance to a query (LLM-as-Judge)."""
    query: str = Field(..., description="The evaluation query text")
    chunk_id: str = Field(..., description="ID of the evaluated chunk")
    chunk_text: str = Field(..., description="Text content of the evaluated chunk")
    retrieval_method: str = Field(..., description="Retriever method that produced this chunk (semantic, recursive, bm25, reranker)")
    relevant: bool = Field(..., description="Whether the LLM judge deemed the chunk relevant to the query")
    reason: str = Field(..., description="Explanation of the relevance judgment")


class EvaluationReport(BaseModel):
    """Aggregated evaluation report containing precision metrics and individual judgments."""
    query: str = Field(..., description="The evaluated user query")
    precision_at_3: Dict[str, float] = Field(..., description="Precision@3 mapping per retrieval strategy (semantic, recursive, bm25, reranker)")
    precision_at_5: Dict[str, float] = Field(..., description="Precision@5 mapping per retrieval strategy (semantic, recursive, bm25, reranker)")
    judgments: List[EvaluationResult] = Field(default_factory=list, description="List of individual chunk judgments")
