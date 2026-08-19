import logging
from typing import Dict, List, Optional

from src.deduplication.chunk_deduplicator import ChunkDeduplicator
from src.embedding.embedding_service import EmbeddingService
from src.evaluation.llm_judge import LLMJudge
from src.models.evaluation import EvaluationReport, EvaluationResult
from src.reranking.bge_reranker import BGEReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.recursive_retriever import RecursiveRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.vectorstore.qdrant_store import QdrantVectorStore

logger = logging.getLogger("medical_rag.evaluation.rag_evaluator")


class RAGEvaluator:
    """Evaluates retrieval quality across semantic, recursive, BM25, and reranker stages using Precision@K metrics."""

    def __init__(
        self,
        semantic_retriever: Optional[SemanticRetriever] = None,
        recursive_retriever: Optional[RecursiveRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        deduplicator: Optional[ChunkDeduplicator] = None,
        reranker: Optional[BGEReranker] = None,
        llm_judge: Optional[LLMJudge] = None
    ):
        embedding_service = EmbeddingService()
        vector_store = QdrantVectorStore()

        self.semantic_retriever = semantic_retriever or SemanticRetriever(
            embedding_service=embedding_service, vector_store=vector_store
        )
        self.recursive_retriever = recursive_retriever or RecursiveRetriever(
            embedding_service=embedding_service, vector_store=vector_store
        )
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.deduplicator = deduplicator or ChunkDeduplicator()
        self.reranker = reranker or BGEReranker()
        self.llm_judge = llm_judge or LLMJudge()

    def evaluate_query(self, query: str, top_k: int = 5) -> EvaluationReport:
        """Runs multi-retrieval evaluation for a single query.

        Args:
            query: User test question text.
            top_k: Top-K context size (e.g. 5).

        Returns:
            EvaluationReport containing P@3, P@5, and individual judgments.
        """
        logger.info(f"Evaluating retrieval quality for query: '{query}' (top_k={top_k})")

        # 1. Retrieve from each individual strategy
        sem_res = self.semantic_retriever.retrieve(query, top_k=top_k)
        rec_res = self.recursive_retriever.retrieve(query, top_k=top_k)
        bm25_res = self.bm25_retriever.retrieve(query, top_k=top_k)

        # 2. Hybrid union -> deduplicate -> rerank
        all_candidates = sem_res + rec_res + bm25_res
        dedup_candidates = self.deduplicator.deduplicate(all_candidates)
        rerank_res = self.reranker.rerank(query, dedup_candidates, top_k=top_k)

        method_chunks = {
            "semantic": [r.chunk for r in sem_res[:top_k]],
            "recursive": [r.chunk for r in rec_res[:top_k]],
            "bm25": [r.chunk for r in bm25_res[:top_k]],
            "reranker": [r.chunk for r in rerank_res[:top_k]],
        }

        all_judgments: List[EvaluationResult] = []
        p3_map: Dict[str, float] = {}
        p5_map: Dict[str, float] = {}

        # 3. Evaluate each retrieval method in isolated LLM batch calls
        for method, chunks in method_chunks.items():
            if not chunks:
                p3_map[method] = 0.0
                p5_map[method] = 0.0
                continue

            # Isolated batch call to LLM for this retrieval method
            judgments_for_method = self.llm_judge.judge_batch_by_method(
                query=query,
                chunks=chunks,
                retrieval_method=method
            )
            all_judgments.extend(judgments_for_method)

            # Compute Precision@3
            p3_chunks = judgments_for_method[:3]
            rel_3 = sum(1 for j in p3_chunks if j.relevant)
            p3_map[method] = round(rel_3 / max(1, len(p3_chunks)), 2) if p3_chunks else 0.0

            # Compute Precision@5
            p5_chunks = judgments_for_method[:5]
            rel_5 = sum(1 for j in p5_chunks if j.relevant)
            p5_map[method] = round(rel_5 / max(1, len(p5_chunks)), 2) if p5_chunks else 0.0

        return EvaluationReport(
            query=query,
            precision_at_3=p3_map,
            precision_at_5=p5_map,
            judgments=all_judgments
        )

    def evaluate_queries(self, queries: List[str], top_k: int = 5) -> List[EvaluationReport]:
        """Runs evaluation over a batch of queries."""
        reports = []
        for q in queries:
            reports.append(self.evaluate_query(q, top_k=top_k))
        return reports
