from enum import Enum
from pydantic import BaseModel, Field

from src.models.chunk import Chunk


class RetrievalMethod(str, Enum):
    """Retrieval strategy enum."""
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    BM25 = "bm25"


class RetrievalResult(BaseModel):
    """A chunk scored for relevance to a user query by a retrieval strategy."""
    chunk: Chunk = Field(..., description="The retrieved chunk entity")
    score: float = Field(..., description="Relevance score returned by retrieval strategy")
    retrieval_method: RetrievalMethod = Field(..., description="Which retriever produced this result: semantic, recursive, or bm25")


class RerankResult(BaseModel):
    """A chunk re-scored by the reranker model after deduplication."""
    chunk: Chunk = Field(..., description="The reranked chunk entity")
    rerank_score: float = Field(..., description="Score returned by cross-encoder reranker model")
    original_retrieval_method: RetrievalMethod = Field(..., description="Which retriever originally found this chunk")
    original_score: float = Field(..., description="Original score before reranking")
