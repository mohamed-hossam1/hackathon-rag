import logging
from typing import List, Optional, Tuple
from sentence_transformers import CrossEncoder

from src.config import get_config
from src.models.retrieval import RerankResult, RetrievalResult

logger = logging.getLogger("medical_rag.reranking")


class BGEReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-base model."""

    def __init__(self, model_name: Optional[str] = None):
        config = get_config()
        self.model_name = model_name or config.RERANKER_MODEL
        self.top_k_default = config.RERANKER_TOP_K
        logger.info(f"Loading CrossEncoder reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Reranks candidate retrieval results against a query string.

        Args:
            query: User search query text.
            candidates: List of RetrievalResult objects to rerank.
            top_k: Optional top-K override. Defaults to config.RERANKER_TOP_K.

        Returns:
            List of RerankResult objects sorted by rerank_score descending.
        """
        if not query.strip() or not candidates:
            return []

        limit = top_k if top_k is not None else self.top_k_default

        # Construct query-chunk text pairs for CrossEncoder scoring (tuples)
        pairs: List[Tuple[str, str]] = [(query, candidate.chunk.text) for candidate in candidates]
        logger.info(f"Reranking {len(pairs)} query-chunk pairs with model {self.model_name}")

        scores = self.model.predict(pairs)

        rerank_results: List[RerankResult] = []
        for candidate, score in zip(candidates, scores):
            rerank_results.append(
                RerankResult(
                    chunk=candidate.chunk,
                    rerank_score=float(score),
                    original_retrieval_method=candidate.retrieval_method,
                    original_score=candidate.score
                )
            )

        # Sort by rerank_score descending
        rerank_results.sort(key=lambda r: r.rerank_score, reverse=True)
        final_results = rerank_results[:limit]
        logger.info(f"Reranker returned top {len(final_results)} reranked results")
        return final_results
