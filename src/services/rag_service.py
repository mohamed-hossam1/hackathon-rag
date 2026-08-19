import logging
import re
from typing import Dict, List, Optional

from src.config import get_config
from src.deduplication.chunk_deduplicator import ChunkDeduplicator
from src.embedding.embedding_service import EmbeddingService
from src.llm.base import LLMService
from src.llm.llm_service import OpenAILikeLLMService
from src.models.chunk import Chunk
from src.models.citation import Citation, CitationValidationResult
from src.models.response import (
    DEFAULT_MEDICAL_DISCLAIMER,
    ConfidenceLabel,
    DevTrace,
    RAGResponse,
)
from src.models.retrieval import RerankResult, RetrievalResult
from src.reranking.bge_reranker import BGEReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.recursive_retriever import RecursiveRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.validation.citation_validator import CitationValidator
from src.vectorstore.qdrant_store import QdrantVectorStore

logger = logging.getLogger("medical_rag.services.rag")

ABSTENTION_MESSAGE = "I couldn't find sufficient evidence in the uploaded documents to answer this question."

RAG_SYSTEM_PROMPT = """You are an expert clinical AI assistant answering medical questions based EXCLUSIVELY on the provided WHO reference context chunks.

STRICT GROUNDING RULES:
1. Answer the question using ONLY information directly stated in the provided Context Chunks.
2. Do NOT use outside medical knowledge, assumptions, or extrapolations.
3. Every factual claim or statement MUST end with an inline citation tag using the exact format:
   [Doc: <filename>, Page: <page_number>, ChunkID: <chunk_id>]
4. If the provided context does NOT contain sufficient evidence to answer the question, state EXACTLY:
   "I couldn't find sufficient evidence in the uploaded documents to answer this question."
5. Never invent or hallucinate citations or chunk IDs."""


class RAGService:
    """Orchestrator service for tri-hybrid retrieval, deduplication, reranking, LLM answer generation, citation validation, and evidence scoring."""

    def __init__(
        self,
        semantic_retriever: Optional[SemanticRetriever] = None,
        recursive_retriever: Optional[RecursiveRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        deduplicator: Optional[ChunkDeduplicator] = None,
        reranker: Optional[BGEReranker] = None,
        llm_service: Optional[LLMService] = None,
        citation_validator: Optional[CitationValidator] = None,
        evidence_threshold: Optional[float] = None
    ):
        config = get_config()
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
        self.llm_service = llm_service or OpenAILikeLLMService()
        self.citation_validator = citation_validator or CitationValidator(
            llm_service=self.llm_service
        )
        self.evidence_threshold = (
            evidence_threshold if evidence_threshold is not None else config.EVIDENCE_THRESHOLD
        )
        self.top_k_retrieval = config.RETRIEVAL_TOP_K
        self.top_k_rerank = config.RERANKER_TOP_K

    def query(self, query_text: str, dev: bool = False) -> RAGResponse:
        """Executes full RAG query pipeline.

        Args:
            query_text: User medical question text.
            dev: If True, attaches detailed step-by-step DevTrace.

        Returns:
            RAGResponse object.
        """
        if not query_text.strip():
            return RAGResponse(
                answer=ABSTENTION_MESSAGE,
                citations=[],
                evidence_score=0.0,
                confidence_label=ConfidenceLabel.INSUFFICIENT,
                abstained=True,
                disclaimer=DEFAULT_MEDICAL_DISCLAIMER
            )

        logger.info(f"Executing RAG query: '{query_text[:50]}...' (dev={dev})")

        # Step 1: Tri-Hybrid Retrieval
        sem_results = self.semantic_retriever.retrieve(query_text, top_k=self.top_k_retrieval)
        rec_results = self.recursive_retriever.retrieve(query_text, top_k=self.top_k_retrieval)
        bm25_results = self.bm25_retriever.retrieve(query_text, top_k=self.top_k_retrieval)

        # Step 2: Union & Deduplication
        all_candidates = sem_results + rec_results + bm25_results
        dedup_candidates = self.deduplicator.deduplicate(all_candidates)

        # Step 3: Reranking
        reranker_results = self.reranker.rerank(query_text, dedup_candidates, top_k=self.top_k_rerank)
        selected_context = reranker_results[:self.top_k_rerank]

        # Check if any context retrieved
        if not selected_context:
            logger.warning("No context chunks found; returning abstention response")
            return self._build_abstention_response(
                dev=dev,
                sem_results=sem_results,
                rec_results=rec_results,
                bm25_results=bm25_results,
                dedup_candidates=dedup_candidates,
                reranker_results=reranker_results,
                selected_context=[]
            )

        # Step 4: Assemble Prompt Context
        context_str_blocks = []
        chunks_map: Dict[str, Chunk] = {}

        for idx, rerank_item in enumerate(selected_context, 1):
            chunk = rerank_item.chunk
            chunks_map[chunk.chunk_id] = chunk
            context_str_blocks.append(
                f"--- Context Chunk #{idx} ---\n"
                f"ChunkID: {chunk.chunk_id}\n"
                f"Filename: {chunk.filename}\n"
                f"Page: {chunk.page_start}\n"
                f"Text:\n{chunk.text}\n"
            )

        context_prompt = "\n".join(context_str_blocks)
        user_prompt = f"Context:\n{context_prompt}\n\nUser Question:\n{query_text}"

        # Step 5: Generate Answer
        raw_answer = self.llm_service.generate(
            prompt=user_prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
            temperature=0.0
        )

        # Step 6: Parse Citations
        citations = self._parse_citations(raw_answer, chunks_map)

        # Step 7: Validate Citations & Evidence Scoring
        validations = self.citation_validator.validate_citations(citations, chunks_map)
        evidence_score = self.citation_validator.compute_evidence_score(validations)
        confidence_label = self.citation_validator.get_confidence_label(evidence_score, validations)

        # Step 8: Abstention Check
        is_abstention_text = ABSTENTION_MESSAGE.lower() in raw_answer.lower()
        if is_abstention_text or evidence_score < self.evidence_threshold:
            logger.info(f"Abstaining: evidence_score={evidence_score:.2f} < threshold={self.evidence_threshold}")
            return self._build_abstention_response(
                dev=dev,
                sem_results=sem_results,
                rec_results=rec_results,
                bm25_results=bm25_results,
                dedup_candidates=dedup_candidates,
                reranker_results=reranker_results,
                selected_context=selected_context,
                validations=validations
            )

        # Clean citation tags from answer text for clean user display if desired, or preserve inline
        dev_trace_obj = None
        if dev:
            dev_trace_obj = DevTrace(
                semantic_results=sem_results,
                recursive_results=rec_results,
                bm25_results=bm25_results,
                deduplicated_candidates=dedup_candidates,
                reranker_results=reranker_results,
                selected_context=selected_context,
                citation_validations=validations
            )

        return RAGResponse(
            answer=raw_answer,
            citations=citations,
            citation_validations=validations if dev else None,
            evidence_score=evidence_score,
            confidence_label=confidence_label,
            abstained=False,
            disclaimer=DEFAULT_MEDICAL_DISCLAIMER,
            dev_trace=dev_trace_obj
        )

    def _parse_citations(self, answer_text: str, chunks_map: Dict[str, Chunk]) -> List[Citation]:
        """Extracts structured Citation objects from inline citation tags in the answer."""
        citations: List[Citation] = []

        # Matches [Doc: filename, Page: page_num, ChunkID: chunk_id]
        citation_pattern = re.compile(
            r"\[Doc:\s*(?P<filename>[^,]+),\s*Page:\s*(?P<page>\d+),\s*ChunkID:\s*(?P<chunk_id>[^\]]+)\]"
        )

        lines = answer_text.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            matches = list(citation_pattern.finditer(line_str))
            if not matches:
                continue

            for match in matches:
                chunk_id = match.group("chunk_id").strip()
                filename = match.group("filename").strip()
                page_start = int(match.group("page"))

                # Extract preceding sentence as claim text
                claim_text = line_str[: match.start()].strip()
                # Clean preceding citation tags from claim text if multiple citations on line
                claim_text = re.sub(r"\[Doc:.*\]", "", claim_text).strip()
                if not claim_text:
                    claim_text = line_str

                chunk = chunks_map.get(chunk_id)
                document_id = chunk.document_id if chunk else "unknown"
                page_end = chunk.page_end if chunk else page_start

                citations.append(
                    Citation(
                        claim=claim_text,
                        chunk_id=chunk_id,
                        document_id=document_id,
                        filename=filename,
                        page_start=page_start,
                        page_end=page_end
                    )
                )

        return citations

    def _build_abstention_response(
        self,
        dev: bool,
        sem_results: List[RetrievalResult],
        rec_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        dedup_candidates: List[RetrievalResult],
        reranker_results: List[RerankResult],
        selected_context: List[RerankResult],
        validations: Optional[List[CitationValidationResult]] = None
    ) -> RAGResponse:
        """Helper to construct an abstention RAGResponse."""
        dev_trace_obj = None
        if dev:
            dev_trace_obj = DevTrace(
                semantic_results=sem_results,
                recursive_results=rec_results,
                bm25_results=bm25_results,
                deduplicated_candidates=dedup_candidates,
                reranker_results=reranker_results,
                selected_context=selected_context,
                citation_validations=validations or []
            )

        return RAGResponse(
            answer=ABSTENTION_MESSAGE,
            citations=[],
            citation_validations=validations if dev else None,
            evidence_score=0.0,
            confidence_label=ConfidenceLabel.INSUFFICIENT,
            abstained=True,
            disclaimer=DEFAULT_MEDICAL_DISCLAIMER,
            dev_trace=dev_trace_obj
        )
