import logging
from typing import Any, Dict, List, Optional
import httpx

from src.config import get_config

logger = logging.getLogger("medical_rag.db.supabase")


class SupabaseService:
    """Service for managing persistence of dev queries, retrieval traces, and AI-Judge evaluation reports in Supabase."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None
    ):
        config = get_config()
        self.url = (supabase_url or config.SUPABASE_URL or "").rstrip("/")
        self.key = supabase_key or config.SUPABASE_SECRET_KEY or config.SUPABASE_KEY or config.SUPABASE_PUBLISHABLE_KEY or ""

    @property
    def is_configured(self) -> bool:
        """Returns True if Supabase URL and Key are provided."""
        return bool(self.url and self.key)

    def _get_headers(self) -> Dict[str, str]:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

    def save_dev_trace(
        self,
        query_text: str,
        abstained: bool,
        evidence_score: float,
        semantic_chunks: List[Any],
        recursive_chunks: List[Any],
        bm25_chunks: List[Any],
        reranker_chunks: List[Any]
    ) -> Optional[str]:
        """Saves dev query metadata and top-10 retrieval trace chunks per strategy to Supabase.

        Returns:
            The generated query_id UUID string, or None if save failed / unconfigured.
        """
        if not self.is_configured:
            logger.warning("Supabase URL or Key not configured; skipping dev trace save")
            return None

        try:
            # 1. Insert dev_query
            query_payload = {
                "query_text": query_text,
                "abstained": abstained,
                "evidence_score": float(evidence_score)
            }

            resp = httpx.post(
                f"{self.url}/rest/v1/dev_queries",
                headers=self._get_headers(),
                json=query_payload,
                timeout=10.0
            )
            resp.raise_for_status()
            created_records = resp.json()
            if not created_records:
                logger.error("Failed to insert dev query into Supabase: empty response")
                return None

            query_id = created_records[0]["id"]

            # 2. Prepare top-10 chunks for each retrieval method
            trace_records: List[Dict[str, Any]] = []

            methods_map = [
                ("semantic", semantic_chunks),
                ("recursive", recursive_chunks),
                ("bm25", bm25_chunks),
                ("reranker", reranker_chunks)
            ]

            for method_name, results_list in methods_map:
                top_10 = results_list[:10]
                for rank_idx, item in enumerate(top_10, 1):
                    # Handle both RetrievalResult/RerankResult objects and dicts/Chunk objects
                    chunk_obj = getattr(item, "chunk", item)
                    score_val = getattr(item, "score", 0.0)

                    chunk_id = getattr(chunk_obj, "chunk_id", str(getattr(chunk_obj, "id", "unknown")))
                    chunk_text = getattr(chunk_obj, "text", str(chunk_obj))
                    filename = getattr(chunk_obj, "filename", None)
                    page_start = getattr(chunk_obj, "page_start", None)

                    trace_records.append({
                        "query_id": query_id,
                        "retrieval_method": method_name,
                        "rank": rank_idx,
                        "chunk_id": chunk_id,
                        "chunk_text": chunk_text,
                        "score": float(score_val),
                        "filename": filename,
                        "page_start": page_start
                    })

            if trace_records:
                trace_resp = httpx.post(
                    f"{self.url}/rest/v1/retrieval_traces",
                    headers=self._get_headers(),
                    json=trace_records,
                    timeout=10.0
                )
                trace_resp.raise_for_status()

            logger.info(f"Saved dev trace for query '{query_text[:30]}...' with ID {query_id} ({len(trace_records)} chunks saved)")
            return query_id

        except Exception as err:
            logger.error(f"Error saving dev trace to Supabase: {err}", exc_info=True)
            return None

    def get_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetches list of saved dev queries from Supabase."""
        if not self.is_configured:
            return []

        try:
            resp = httpx.get(
                f"{self.url}/rest/v1/dev_queries?select=*&order=created_at.desc&limit={limit}",
                headers=self._get_headers(),
                timeout=10.0
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as err:
            logger.error(f"Error fetching dev queries from Supabase: {err}")
            return []

    def get_query_traces(self, query_id: str) -> Dict[str, Any]:
        """Fetches saved retrieval traces for a specific query_id, grouped by retrieval_method."""
        if not self.is_configured:
            return {"query_id": query_id, "traces": {}}

        try:
            # Get query detail
            q_resp = httpx.get(
                f"{self.url}/rest/v1/dev_queries?id=eq.{query_id}&select=*",
                headers=self._get_headers(),
                timeout=10.0
            )
            q_resp.raise_for_status()
            queries = q_resp.json()
            query_detail = queries[0] if queries else {}

            # Get traces
            t_resp = httpx.get(
                f"{self.url}/rest/v1/retrieval_traces?query_id=eq.{query_id}&select=*&order=rank.asc",
                headers=self._get_headers(),
                timeout=10.0
            )
            t_resp.raise_for_status()
            rows = t_resp.json()

            grouped_traces: Dict[str, List[Dict[str, Any]]] = {
                "semantic": [],
                "recursive": [],
                "bm25": [],
                "reranker": []
            }

            for row in rows:
                method = row.get("retrieval_method")
                if method in grouped_traces:
                    grouped_traces[method].append({
                        "rank": row.get("rank"),
                        "chunk_id": row.get("chunk_id"),
                        "chunk_text": row.get("chunk_text"),
                        "score": row.get("score"),
                        "filename": row.get("filename"),
                        "page_start": row.get("page_start")
                    })

            return {
                "query_id": query_id,
                "query_text": query_detail.get("query_text", ""),
                "created_at": query_detail.get("created_at"),
                "abstained": query_detail.get("abstained", False),
                "evidence_score": query_detail.get("evidence_score", 0.0),
                "traces": grouped_traces
            }
        except Exception as err:
            logger.error(f"Error fetching traces for query_id={query_id}: {err}")
            return {"query_id": query_id, "traces": {}}

    def save_evaluation_report(
        self,
        query_id: str,
        retrieval_method: str,
        precision_at_3: float,
        precision_at_5: float,
        judgments: List[Dict[str, Any]]
    ) -> bool:
        """Saves an AI-Judge evaluation report row for a query method to Supabase."""
        if not self.is_configured:
            return False

        try:
            payload = {
                "query_id": query_id,
                "retrieval_method": retrieval_method,
                "precision_at_3": float(precision_at_3),
                "precision_at_5": float(precision_at_5),
                "judgments": judgments
            }

            resp = httpx.post(
                f"{self.url}/rest/v1/evaluation_reports",
                headers=self._get_headers(),
                json=payload,
                timeout=10.0
            )
            resp.raise_for_status()
            return True
        except Exception as err:
            logger.error(f"Error saving evaluation report to Supabase: {err}")
            return False

    def get_evaluation_reports(self, query_id: str) -> List[Dict[str, Any]]:
        """Fetches existing evaluation reports for a query_id from Supabase."""
        if not self.is_configured:
            return []

        try:
            resp = httpx.get(
                f"{self.url}/rest/v1/evaluation_reports?query_id=eq.{query_id}&select=*",
                headers=self._get_headers(),
                timeout=10.0
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as err:
            logger.error(f"Error fetching evaluation reports for query_id={query_id}: {err}")
            return []
