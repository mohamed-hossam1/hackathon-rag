import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any

from src.config import get_config
from src.db.supabase_service import SupabaseService
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
from src.services.memory_detector import PersonalMemoryDetector
from src.validation.citation_validator import CitationValidator
from src.vectorstore.qdrant_store import QdrantVectorStore

logger = logging.getLogger("medical_rag.services.rag")

ABSTENTION_MESSAGE = "I couldn't find sufficient evidence in the uploaded documents to answer this question."

RAG_SYSTEM_PROMPT = """You are an expert clinical AI assistant answering medical questions based EXCLUSIVELY on the provided WHO reference context chunks.

STRICT GROUNDING RULES:
1. Answer the question using ONLY information directly stated in the provided Context Chunks.
2. Do NOT use outside medical knowledge, assumptions, or extrapolations.
3. Every factual claim or statement MUST end with an inline citation tag placed IMMEDIATELY after the specific sentence it supports using the exact format:
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
        evidence_threshold: Optional[float] = None,
        supabase_service: Optional[SupabaseService] = None
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
        self.supabase_service = supabase_service or SupabaseService()
        self.evidence_threshold = (
            evidence_threshold if evidence_threshold is not None else config.EVIDENCE_THRESHOLD
        )
        self.top_k_retrieval = config.RETRIEVAL_TOP_K
        self.top_k_rerank = config.RERANKER_TOP_K
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag_parallel")        
        self.memory_detector = PersonalMemoryDetector(llm_service=self.llm_service)

    def _parallel_retrieve(
        self, query_text: str
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult], List[RetrievalResult]]:
        """Executes semantic, recursive, and BM25 retrievals concurrently using a thread pool."""
        fut_sem = self._executor.submit(self.semantic_retriever.retrieve, query_text, self.top_k_retrieval)
        fut_rec = self._executor.submit(self.recursive_retriever.retrieve, query_text, self.top_k_retrieval)
        fut_bm25 = self._executor.submit(self.bm25_retriever.retrieve, query_text, self.top_k_retrieval)

        sem_results = fut_sem.result()
        rec_results = fut_rec.result()
        bm25_results = fut_bm25.result()

        return sem_results, rec_results, bm25_results

    def _async_save_dev_trace(self, background_tasks: Optional[Any] = None, **kwargs) -> None:
        """Asynchronously dispatches dev trace saving to Supabase via FastAPI BackgroundTasks or ThreadPoolExecutor."""
        if background_tasks is not None and hasattr(background_tasks, "add_task"):
            background_tasks.add_task(self.supabase_service.save_dev_trace, **kwargs)
        else:
            self._executor.submit(self.supabase_service.save_dev_trace, **kwargs)

    def _detect_personal_info(self, query_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Detects if user query contains personal medical history, condition, allergies, or medications.

        Returns:
            Tuple of (has_personal_info, extracted_personal_info, memory_prompt)
        """
        text = query_text.strip()
        if not text:
            return False, None, None

        return self.memory_detector.detect(query_text)

    def query(
        self,
        query_text: str,
        dev: bool = False,
        personal_context: Optional[str] = None,
        user_id: Optional[str] = None,
        background_tasks: Optional[Any] = None
    ) -> RAGResponse:
        """Executes full RAG query pipeline.

        Args:
            query_text: User medical question text.
            dev: If True, attaches detailed step-by-step DevTrace.
            personal_context: Saved user personal memory/context text.
            user_id: Optional authenticated user ID for auto-fetching saved memories.

        Returns:
            RAGResponse object.
        """
        has_p_info, extracted_p_info, mem_prompt = self._detect_personal_info(query_text)

        if not query_text.strip():
            return RAGResponse(
                answer=ABSTENTION_MESSAGE,
                citations=[],
                evidence_score=0.0,
                confidence_label=ConfidenceLabel.INSUFFICIENT,
                abstained=True,
                disclaimer=DEFAULT_MEDICAL_DISCLAIMER,
                has_personal_info=has_p_info,
                extracted_personal_info=extracted_p_info,
                memory_prompt=mem_prompt
            )

        # Auto-hydrate user memories if user_id is passed and personal_context is empty
        if user_id and not personal_context:
            memories = self.supabase_service.get_user_memories(user_id)
            if memories:
                m_texts = [f"- {m.get('memory_text', '').strip()}" for m in memories if m.get('memory_text', '').strip()]
                if m_texts:
                    personal_context = "\n".join(m_texts)

        logger.info(f"Executing RAG query: '{query_text[:50]}...' (dev={dev}, personal_context={bool(personal_context)})")

        # Step 1: Tri-Hybrid Retrieval (Parallelized via ThreadPoolExecutor)
        sem_results, rec_results, bm25_results = self._parallel_retrieve(query_text)

        # Step 2: Combine candidates using Reciprocal Rank Fusion (RRF) & Deduplicate
        all_candidates = self._combine_candidates_with_rrf(sem_results, rec_results, bm25_results)
        dedup_candidates = self.deduplicator.deduplicate(all_candidates)

        # Step 3: Reranking
        reranker_results = self.reranker.rerank(query_text, dedup_candidates, top_k=self.top_k_rerank)
        selected_context = reranker_results[:self.top_k_rerank]

        # Check if any context retrieved
        if not selected_context:
            logger.warning("No context chunks found; returning abstention response")
            return self._build_abstention_response(
                dev=dev,
                query_text=query_text,
                sem_results=sem_results,
                rec_results=rec_results,
                bm25_results=bm25_results,
                dedup_candidates=dedup_candidates,
                reranker_results=reranker_results,
                selected_context=[],
                background_tasks=background_tasks
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
        if personal_context and personal_context.strip():
            user_prompt = f"User Personal Context & Medical Profile:\n{personal_context.strip()}\n\nContext:\n{context_prompt}\n\nUser Question:\n{query_text}"
        else:
            user_prompt = f"Context:\n{context_prompt}\n\nUser Question:\n{query_text}"

        # Step 5: Generate Answer via streaming
        raw_answer = "".join(
            self.llm_service.generate_stream(
                prompt=user_prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
                temperature=0.0
            )
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
                query_text=query_text,
                sem_results=sem_results,
                rec_results=rec_results,
                bm25_results=bm25_results,
                dedup_candidates=dedup_candidates,
                reranker_results=reranker_results,
                selected_context=selected_context,
                validations=validations,
                background_tasks=background_tasks
            )

        # Clean citation tags from answer text for clean user display if desired, or preserve inline
        dev_trace_obj = None
        if dev:
            self._async_save_dev_trace(
                background_tasks=background_tasks,
                query_text=query_text,
                abstained=False,
                evidence_score=evidence_score,
                semantic_chunks=sem_results,
                recursive_chunks=rec_results,
                bm25_chunks=bm25_results,
                reranker_chunks=reranker_results
            )
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
            citation_validations=validations,
            evidence_score=evidence_score,
            confidence_label=confidence_label,
            abstained=False,
            disclaimer=DEFAULT_MEDICAL_DISCLAIMER,
            has_personal_info=has_p_info,
            extracted_personal_info=extracted_p_info,
            memory_prompt=mem_prompt,
            dev_trace=dev_trace_obj
        )

    async def query_stream(
        self,
        query_text: str,
        dev: bool = False,
        personal_context: Optional[str] = None,
        user_id: Optional[str] = None,
        background_tasks: Optional[Any] = None
    ):
        """Executes full RAG query pipeline and yields SSE events as answer is generated."""
        has_p_info, extracted_p_info, mem_prompt = self._detect_personal_info(query_text)

        if not query_text.strip():
            resp = RAGResponse(
                answer=ABSTENTION_MESSAGE,
                citations=[],
                evidence_score=0.0,
                confidence_label=ConfidenceLabel.INSUFFICIENT,
                abstained=True,
                disclaimer=DEFAULT_MEDICAL_DISCLAIMER,
                has_personal_info=has_p_info,
                extracted_personal_info=extracted_p_info,
                memory_prompt=mem_prompt
            )
            yield f"event: final\ndata: {resp.model_dump_json()}\n\n"
            return

        # Auto-hydrate user memories if user_id is passed and personal_context is empty
        if user_id and not personal_context:
            memories = self.supabase_service.get_user_memories(user_id)
            if memories:
                m_texts = [f"- {m.get('memory_text', '').strip()}" for m in memories if m.get('memory_text', '').strip()]
                if m_texts:
                    personal_context = "\n".join(m_texts)

        logger.info(f"Executing RAG query stream: '{query_text[:50]}...' (dev={dev}, personal_context={bool(personal_context)})")

        # Step 1: Tri-Hybrid Retrieval (Parallelized via ThreadPoolExecutor)
        sem_results, rec_results, bm25_results = self._parallel_retrieve(query_text)

        # Step 2: Combine candidates using Reciprocal Rank Fusion (RRF) & Deduplicate
        all_candidates = self._combine_candidates_with_rrf(sem_results, rec_results, bm25_results)
        dedup_candidates = self.deduplicator.deduplicate(all_candidates)

        # Step 3: Reranking
        reranker_results = self.reranker.rerank(query_text, dedup_candidates, top_k=self.top_k_rerank)
        selected_context = reranker_results[:self.top_k_rerank]

        if not selected_context:
            logger.warning("No context chunks found; yielding abstention response")
            abstain_resp = self._build_abstention_response(
                dev=dev,
                query_text=query_text,
                sem_results=sem_results,
                rec_results=rec_results,
                bm25_results=bm25_results,
                dedup_candidates=dedup_candidates,
                reranker_results=reranker_results,
                selected_context=[],
                background_tasks=background_tasks
            )
            yield f"event: final\ndata: {abstain_resp.model_dump_json()}\n\n"
            return

        # Initial SSE Event: metadata
        metadata_event = {
            "selected_chunks_count": len(selected_context),
            "status": "context_retrieved"
        }
        yield f"event: metadata\ndata: {json.dumps(metadata_event)}\n\n"

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
        if personal_context and personal_context.strip():
            user_prompt = f"User Personal Context & Medical Profile:\n{personal_context.strip()}\n\nContext:\n{context_prompt}\n\nUser Question:\n{query_text}"
        else:
            user_prompt = f"Context:\n{context_prompt}\n\nUser Question:\n{query_text}"

        try:
            # Step 5: Stream LLM Generation
            raw_answer_chunks = []
            for token in self.llm_service.generate_stream(
                prompt=user_prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
                temperature=0.0
            ):
                raw_answer_chunks.append(token)
                token_event = {"delta": token}
                yield f"event: token\ndata: {json.dumps(token_event)}\n\n"
            raw_answer = "".join(raw_answer_chunks)

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
                abstain_resp = self._build_abstention_response(
                    dev=dev,
                    query_text=query_text,
                    sem_results=sem_results,
                    rec_results=rec_results,
                    bm25_results=bm25_results,
                    dedup_candidates=dedup_candidates,
                    reranker_results=reranker_results,
                    selected_context=selected_context,
                    validations=validations,
                    background_tasks=background_tasks
                )
                yield f"event: final\ndata: {abstain_resp.model_dump_json()}\n\n"
                return

            dev_trace_obj = None
            if dev:
                self._async_save_dev_trace(
                    background_tasks=background_tasks,
                    query_text=query_text,
                    abstained=False,
                    evidence_score=evidence_score,
                    semantic_chunks=sem_results,
                    recursive_chunks=rec_results,
                    bm25_chunks=bm25_results,
                    reranker_chunks=reranker_results
                )
                dev_trace_obj = DevTrace(
                    semantic_results=sem_results,
                    recursive_results=rec_results,
                    bm25_results=bm25_results,
                    deduplicated_candidates=dedup_candidates,
                    reranker_results=reranker_results,
                    selected_context=selected_context,
                    citation_validations=validations
                )

            final_response = RAGResponse(
                answer=raw_answer,
                citations=citations,
                citation_validations=validations,
                evidence_score=evidence_score,
                confidence_label=confidence_label,
                abstained=False,
                disclaimer=DEFAULT_MEDICAL_DISCLAIMER,
                has_personal_info=has_p_info,
                extracted_personal_info=extracted_p_info,
                memory_prompt=mem_prompt,
                dev_trace=dev_trace_obj
            )

            yield f"event: final\ndata: {final_response.model_dump_json()}\n\n"
        except Exception as exc:
            logger.error(f"Error during streaming RAG generation: {exc}", exc_info=True)
            err_event = {"error": str(exc), "message": "Streaming generation failed"}
            yield f"event: error\ndata: {json.dumps(err_event)}\n\n"


    @staticmethod
    def _combine_candidates_with_rrf(
        sem_results: List[RetrievalResult],
        rec_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        k: int = 60
    ) -> List[RetrievalResult]:
        """Combines multi-retriever candidate lists using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1.0 / (k + rank)) across each retriever where a chunk appears.
        Preserves original raw retrieval scores on candidate objects for accurate dev tracing.
        """
        combined_map: Dict[str, RetrievalResult] = {}
        rrf_scores: Dict[str, float] = {}

        for results in [sem_results, rec_results, bm25_results]:
            for rank, res in enumerate(results, start=1):
                cid = res.chunk.chunk_id
                rrf_increment = 1.0 / (k + rank)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_increment

                if cid not in combined_map:
                    combined_map[cid] = RetrievalResult(
                        chunk=res.chunk,
                        score=res.score,
                        retrieval_method=res.retrieval_method
                    )

        all_candidates = list(combined_map.values())
        for candidate in all_candidates:
            setattr(candidate, "_fusion_score", rrf_scores.get(candidate.chunk.chunk_id, 0.0))

        def _get_fusion_sort_key(c: RetrievalResult) -> float:
            fusion_val = getattr(c, "_fusion_score", None)
            if isinstance(fusion_val, (int, float)):
                return float(fusion_val)
            return float(c.score)

        all_candidates.sort(key=_get_fusion_sort_key, reverse=True)
        return all_candidates

    @staticmethod
    def _normalize_scores(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Deprecated min-max helper kept for backward compatibility (returns unchanged scores)."""
        return results

    def _parse_citations(self, answer_text: str, chunks_map: Dict[str, Chunk]) -> List[Citation]:
        """Extracts structured Citation objects from inline citation tags in the answer."""
        citations: List[Citation] = []

        # Matches [Doc: filename, Page: page_num, ChunkID: chunk_id] or variations with quotes/brackets
        citation_pattern = re.compile(
            r"[\[【]Doc:\s*[\"']?(?P<filename>[^,\"'\n\]】]+)[\"']?,\s*Page:\s*(?P<page>\d+),\s*ChunkID:\s*[\"']?(?P<chunk_id>[^\"'\s\]】]+)[\"']?[\]】]",
            re.IGNORECASE
        )

        lines = answer_text.split("\n")
        for line_idx, line in enumerate(lines):
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

                # Extract preceding text on same line if present
                raw_claim = line_str[: match.start()].strip()
                raw_claim = re.sub(r"[\[【]Doc:.*?[\]】]", "", raw_claim).strip()

                chunk = chunks_map.get(chunk_id)
                document_id = chunk.document_id if chunk else "unknown"
                page_end = chunk.page_end if chunk else page_start

                # Look back at previous line if raw_claim is empty
                prev_line_claim = None
                if not raw_claim or len(raw_claim.split()) < 4:
                    prev_lines = [
                        l.strip() for l in lines[:line_idx]
                        if l.strip() and not citation_pattern.fullmatch(l.strip())
                    ]
                    if prev_lines:
                        prev_line_claim = re.sub(r"[\[【]Doc:.*?[\]】]", "", prev_lines[-1]).strip()

                candidate_claim = raw_claim if (raw_claim and len(raw_claim.split()) >= 4) else prev_line_claim

                # Smart Clause Matching: Split compound sentences by semicolons, periods, or conjunctions (and that, while, but) to pair chunk with its exact supported sub-claim
                claim_text = candidate_claim or line_str
                if chunk and chunk.text:
                    search_text = candidate_claim if candidate_claim else answer_text
                    clean_search = re.sub(r"[\[【]Doc:.*?[\]】]", "", search_text)
                    raw_clauses = [
                        s.strip() for s in re.split(r"\n+|(?:(?<!\b\d)[;;\.](?!\d\b)\s*)|(?:\s+and\s+that\s+|\s+while\s+|\s+whereas\s+|\s+but\s+)", clean_search)
                        if len(s.strip().split()) >= 4
                    ]
                    if raw_clauses:
                        chunk_words = set(re.findall(r"\w+", chunk.text.lower()))
                        best_clause, best_overlap = None, 0.0
                        for c_cand in raw_clauses:
                            c_words = set(re.findall(r"\w+", c_cand.lower()))
                            if c_words:
                                overlap = len(chunk_words.intersection(c_words)) / len(c_words)
                                if overlap > best_overlap:
                                    best_overlap, best_clause = overlap, c_cand
                        if best_clause and best_overlap >= 0.20:
                            claim_text = best_clause

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
        query_text: str,
        sem_results: List[RetrievalResult],
        rec_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        dedup_candidates: List[RetrievalResult],
        reranker_results: List[RerankResult],
        selected_context: List[RerankResult],
        validations: Optional[List[CitationValidationResult]] = None,
        background_tasks: Optional[Any] = None
    ) -> RAGResponse:
        """Helper to construct an abstention RAGResponse."""
        has_p_info, extracted_p_info, mem_prompt = self._detect_personal_info(query_text)
        dev_trace_obj = None
        if dev:
            self._async_save_dev_trace(
                background_tasks=background_tasks,
                query_text=query_text,
                abstained=True,
                evidence_score=0.0,
                semantic_chunks=sem_results,
                recursive_chunks=rec_results,
                bm25_chunks=bm25_results,
                reranker_chunks=reranker_results
            )
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
            citation_validations=validations,
            evidence_score=0.0,
            confidence_label=ConfidenceLabel.INSUFFICIENT,
            abstained=True,
            disclaimer=DEFAULT_MEDICAL_DISCLAIMER,
            has_personal_info=has_p_info,
            extracted_personal_info=extracted_p_info,
            memory_prompt=mem_prompt,
            dev_trace=dev_trace_obj
        )
