import logging
from typing import Any, Dict, List, Optional
import httpx

from src.config import get_config

logger = logging.getLogger("medical_rag.db.supabase")

_IN_MEMORY_MEMORIES: Dict[str, List[Dict[str, Any]]] = {}
_IN_MEMORY_USERS: Dict[str, Dict[str, Any]] = {}


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
                    score_val = getattr(item, "rerank_score", getattr(item, "score", 0.0))

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

    def save_document(self, document_dict: Dict[str, Any]) -> bool:
        """Saves a document metadata record into Supabase."""
        if not self.is_configured:
            return False

        try:
            headers = self._get_headers()
            headers["Prefer"] = "resolution=merge-duplicates"
            resp = httpx.post(
                f"{self.url}/rest/v1/documents",
                headers=headers,
                json=document_dict,
                timeout=10.0
            )
            resp.raise_for_status()
            logger.info(f"Saved document '{document_dict.get('document_id')}' to Supabase")
            return True
        except Exception as err:
            logger.error(f"Error saving document to Supabase: {err}")
            return False

    def update_document_status(
        self,
        document_id: str,
        status: str,
        total_pages: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Updates document status, total pages, and error message in Supabase."""
        if not self.is_configured:
            return False

        try:
            payload: Dict[str, Any] = {"status": status}
            if total_pages is not None:
                payload["total_pages"] = total_pages
            if error_message is not None:
                payload["error_message"] = error_message

            resp = httpx.patch(
                f"{self.url}/rest/v1/documents?document_id=eq.{document_id}",
                headers=self._get_headers(),
                json=payload,
                timeout=10.0
            )
            resp.raise_for_status()
            logger.info(f"Updated status of document '{document_id}' to '{status}' in Supabase")
            return True
        except Exception as err:
            logger.error(f"Error updating document status in Supabase: {err}")
            return False

    def save_chunks(self, chunks_dicts: List[Dict[str, Any]]) -> bool:
        """Saves a batch of chunk records into Supabase."""
        if not self.is_configured or not chunks_dicts:
            return False

        try:
            headers = self._get_headers()
            headers["Prefer"] = "resolution=merge-duplicates"
            resp = httpx.post(
                f"{self.url}/rest/v1/document_chunks",
                headers=headers,
                json=chunks_dicts,
                timeout=15.0
            )
            resp.raise_for_status()
            logger.info(f"Saved {len(chunks_dicts)} chunks to Supabase")
            return True
        except Exception as err:
            logger.error(f"Error saving chunks to Supabase: {err}")
            return False

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Fetches all documents from Supabase ordered by upload_timestamp descending."""
        if not self.is_configured:
            return []

        try:
            resp = httpx.get(
                f"{self.url}/rest/v1/documents?select=*&order=upload_timestamp.desc",
                headers=self._get_headers(),
                timeout=10.0
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as err:
            logger.error(f"Error fetching all documents from Supabase: {err}")
            return []

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Fetches all document chunks from Supabase for BM25 re-hydration."""
        if not self.is_configured:
            return []

        try:
            resp = httpx.get(
                f"{self.url}/rest/v1/document_chunks?select=*&order=chunk_index.asc",
                headers=self._get_headers(),
                timeout=15.0
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as err:
            logger.error(f"Error fetching all chunks from Supabase: {err}")
            return []

    # =========================================================================
    # Supabase Authentication & User Memory Methods
    # =========================================================================

    def signup_user(self, email: str, password: str, full_name: Optional[str] = None) -> Dict[str, Any]:
        """Signs up a new user via Supabase Auth API endpoint /auth/v1/signup with dev fallback."""
        import uuid
        user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, email.lower()))
        token = f"dev_token_{user_id}_{email.lower()}"
        user_obj = {
            "id": user_id,
            "email": email.lower(),
            "user_metadata": {"full_name": full_name or email.split("@")[0].title()}
        }

        if self.is_configured:
            payload: Dict[str, Any] = {
                "email": email,
                "password": password
            }
            if full_name:
                payload["data"] = {"full_name": full_name}

            headers = {
                "apikey": self.key,
                "Content-Type": "application/json"
            }

            try:
                resp = httpx.post(
                    f"{self.url}/auth/v1/signup",
                    headers=headers,
                    json=payload,
                    timeout=10.0
                )
                if resp.status_code < 400:
                    return resp.json()
                
                err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                msg = err_data.get("msg") or err_data.get("error_description") or err_data.get("message") or "Registration failed"
                if "unregistered api key" not in msg.lower() and "invalid api key" not in msg.lower():
                    logger.error(f"Supabase signup error: {msg}")
                    raise ValueError(msg)
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise exc

        # Dev Fallback mode when Supabase is unconfigured or key is unregistered
        logger.warning("Using local dev authentication fallback for signup.")
        _IN_MEMORY_USERS[email.lower()] = {"password": password, "user": user_obj}
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user_obj
        }

    def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """Logs in an existing user via Supabase Auth API endpoint /auth/v1/token?grant_type=password with dev fallback."""
        import uuid
        user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, email.lower()))
        token = f"dev_token_{user_id}_{email.lower()}"
        user_obj = _IN_MEMORY_USERS.get(email.lower(), {}).get("user") or {
            "id": user_id,
            "email": email.lower(),
            "user_metadata": {"full_name": email.split("@")[0].title()}
        }

        if self.is_configured:
            payload = {
                "email": email,
                "password": password
            }
            headers = {
                "apikey": self.key,
                "Content-Type": "application/json"
            }

            try:
                resp = httpx.post(
                    f"{self.url}/auth/v1/token?grant_type=password",
                    headers=headers,
                    json=payload,
                    timeout=10.0
                )
                if resp.status_code < 400:
                    return resp.json()

                err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                msg = err_data.get("error_description") or err_data.get("msg") or err_data.get("message") or "Invalid email or password"
                if "unregistered api key" not in msg.lower() and "invalid api key" not in msg.lower():
                    logger.error(f"Supabase login error: {msg}")
                    raise ValueError(msg)
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise exc

        # Dev Fallback mode when Supabase is unconfigured or key is unregistered
        if email.lower() in _IN_MEMORY_USERS:
            stored = _IN_MEMORY_USERS[email.lower()]
            if stored.get("password") != password:
                raise ValueError("Invalid email or password")

        logger.warning("Using local dev authentication fallback for login.")
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user_obj
        }

    def get_user_from_token(self, token: str) -> Dict[str, Any]:
        """Validates Bearer token and returns user profile data with dev fallback."""
        if token.startswith("dev_token_"):
            parts = token.split("_", 3)
            u_id = parts[2] if len(parts) > 2 else "dev-user-id"
            u_email = parts[3] if len(parts) > 3 else "user@example.com"
            return {
                "id": u_id,
                "email": u_email,
                "user_metadata": {"full_name": u_email.split("@")[0].title()}
            }

        if not self.is_configured:
            raise RuntimeError("Supabase credentials not configured in environment")

        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {token}"
        }

        resp = httpx.get(
            f"{self.url}/auth/v1/user",
            headers=headers,
            timeout=10.0
        )
        if resp.status_code >= 400:
            raise ValueError("Invalid or expired authentication token")

        return resp.json()

    def save_user_memory(self, user_id: str, memory_text: str) -> Dict[str, Any]:
        """Saves a personal context memory item for a user with dev fallback."""
        import uuid
        import datetime
        mem_item = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "memory_text": memory_text.strip(),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        if self.is_configured:
            headers = self._get_headers()
            headers["Prefer"] = "return=representation"
            payload = {
                "user_id": user_id,
                "memory_text": memory_text.strip()
            }

            try:
                resp = httpx.post(
                    f"{self.url}/rest/v1/user_memories",
                    headers=headers,
                    json=payload,
                    timeout=10.0
                )
                if resp.status_code < 400:
                    created = resp.json()
                    if isinstance(created, list) and created:
                        return created[0]
                    return created
            except Exception as exc:
                logger.warning(f"Failed to save user memory to Supabase: {exc}. Saving locally.")

        if user_id not in _IN_MEMORY_MEMORIES:
            _IN_MEMORY_MEMORIES[user_id] = []
        _IN_MEMORY_MEMORIES[user_id].insert(0, mem_item)
        return mem_item

    def get_user_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieves all saved personal context memory items for a specific user with dev fallback."""
        if self.is_configured:
            try:
                resp = httpx.get(
                    f"{self.url}/rest/v1/user_memories?user_id=eq.{user_id}&select=*&order=created_at.desc",
                    headers=self._get_headers(),
                    timeout=10.0
                )
                if resp.status_code < 400:
                    return resp.json()
            except Exception as err:
                logger.error(f"Error fetching user memories for user_id={user_id}: {err}")

        return _IN_MEMORY_MEMORIES.get(user_id, [])

    def delete_user_memory(self, user_id: str, memory_id: str) -> bool:
        """Deletes a specific user memory item with dev fallback."""
        if self.is_configured:
            try:
                resp = httpx.delete(
                    f"{self.url}/rest/v1/user_memories?id=eq.{memory_id}&user_id=eq.{user_id}",
                    headers=self._get_headers(),
                    timeout=10.0
                )
                if resp.status_code < 400:
                    return True
            except Exception as err:
                logger.error(f"Error deleting user memory id={memory_id}: {err}")

        if user_id in _IN_MEMORY_MEMORIES:
            _IN_MEMORY_MEMORIES[user_id] = [m for m in _IN_MEMORY_MEMORIES[user_id] if m.get("id") != memory_id]
            return True
        return False

    # =========================================================================
    # Chat & Message Persistence Methods
    # =========================================================================

    def create_chat(self, user_id: str, title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Creates a new chat record for a user. Returns the created chat record or None."""
        import uuid
        chat_obj = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "title": (title or "Untitled Chat").strip(),
            "pinned": False,
            "created_at": None,
            "last_message_at": None,
            "deleted": False,
        }

        if self.is_configured:
            try:
                headers = self._get_headers()
                headers["Prefer"] = "return=representation"
                payload = {
                    "user_id": user_id,
                    "title": chat_obj["title"],
                    "pinned": False
                }
                resp = httpx.post(
                    f"{self.url}/rest/v1/chats",
                    headers=headers,
                    json=payload,
                    timeout=10.0
                )
                resp.raise_for_status()
                created = resp.json()
                if isinstance(created, list) and created:
                    return created[0]
                return created
            except Exception as exc:
                logger.warning(f"Failed to create chat in Supabase: {exc}. Falling back to in-memory.")

        # Dev fallback
        if user_id not in _IN_MEMORY_MEMORIES:
            _IN_MEMORY_MEMORIES[user_id] = []
        _IN_MEMORY_MEMORIES[user_id].insert(0, chat_obj)
        return chat_obj

    def list_user_chats(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Lists non-deleted chats for a user ordered by pinned desc, last_message_at desc, created_at desc."""
        if self.is_configured:
            try:
                resp = httpx.get(
                    f"{self.url}/rest/v1/chats?user_id=eq.{user_id}&deleted=eq.false&select=*&order=pinned.desc,last_message_at.desc,created_at.desc&limit={limit}",
                    headers=self._get_headers(),
                    timeout=10.0
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                logger.error(f"Error listing chats for user_id={user_id}: {exc}")

        # Dev fallback: return any in-memory chats for user (stored in user memories map)
        # We store chats in _IN_MEMORY_MEMORIES[user_id] prefix for fallback convenience.
        raw = _IN_MEMORY_MEMORIES.get(user_id, [])
        # Filter objects that look like chats (have 'title')
        chats = [c for c in raw if isinstance(c, dict) and c.get("title") is not None]
        return chats[:limit]

    def add_message(self, chat_id: str, sender: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Adds a message to a chat and updates the chat's last_message_at timestamp."""
        import uuid, datetime
        message_obj = {
            "id": str(uuid.uuid4()),
            "chat_id": chat_id,
            "sender": sender,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        if self.is_configured:
            try:
                headers = self._get_headers()
                headers["Prefer"] = "return=representation"
                payload = {
                    "chat_id": chat_id,
                    "sender": sender,
                    "content": content,
                    "metadata": metadata or {}
                }
                resp = httpx.post(
                    f"{self.url}/rest/v1/chat_messages",
                    headers=headers,
                    json=payload,
                    timeout=10.0
                )
                resp.raise_for_status()
                created = resp.json()

                # Update chat last_message_at
                try:
                    patch_payload = {"last_message_at": message_obj["created_at"]}
                    httpx.patch(
                        f"{self.url}/rest/v1/chats?id=eq.{chat_id}",
                        headers=self._get_headers(),
                        json=patch_payload,
                        timeout=5.0
                    )
                except Exception:
                    pass

                if isinstance(created, list) and created:
                    return created[0]
                return created
            except Exception as exc:
                logger.error(f"Failed to add message to Supabase for chat_id={chat_id}: {exc}")

        # Dev fallback: store in-memory under a special key
        key = f"_chat_msgs_{chat_id}"
        if key not in _IN_MEMORY_MEMORIES:
            _IN_MEMORY_MEMORIES[key] = []
        _IN_MEMORY_MEMORIES[key].append(message_obj)

        # Also attempt to update in-memory chat last_message_at
        for u, arr in _IN_MEMORY_MEMORIES.items():
            for item in arr:
                if isinstance(item, dict) and item.get("id") == chat_id:
                    item["last_message_at"] = message_obj["created_at"]
        return message_obj

    def get_chat_messages(self, chat_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieves messages for a chat ordered by created_at ascending."""
        if self.is_configured:
            try:
                resp = httpx.get(
                    f"{self.url}/rest/v1/chat_messages?chat_id=eq.{chat_id}&select=*&order=created_at.asc&limit={limit}&offset={offset}",
                    headers=self._get_headers(),
                    timeout=10.0
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                logger.error(f"Error fetching messages for chat_id={chat_id}: {exc}")

        key = f"_chat_msgs_{chat_id}"
        return _IN_MEMORY_MEMORIES.get(key, [])[offset: offset + limit]

    def soft_delete_chat(self, chat_id: str) -> bool:
        """Soft-deletes a chat (sets deleted=true)."""
        if self.is_configured:
            try:
                resp = httpx.patch(
                    f"{self.url}/rest/v1/chats?id=eq.{chat_id}",
                    headers=self._get_headers(),
                    json={"deleted": True},
                    timeout=10.0
                )
                resp.raise_for_status()
                return True
            except Exception as exc:
                logger.error(f"Error soft-deleting chat_id={chat_id}: {exc}")
                return False

        # Dev fallback: remove from in-memory list
        removed = False
        for u, arr in list(_IN_MEMORY_MEMORIES.items()):
            new_arr = [i for i in arr if not (isinstance(i, dict) and i.get("id") == chat_id)]
            if len(new_arr) != len(arr):
                _IN_MEMORY_MEMORIES[u] = new_arr
                removed = True
        return removed

    def pin_chat(self, chat_id: str, pinned: bool = True) -> bool:
        """Sets the pinned flag on a chat."""
        if self.is_configured:
            try:
                resp = httpx.patch(
                    f"{self.url}/rest/v1/chats?id=eq.{chat_id}",
                    headers=self._get_headers(),
                    json={"pinned": bool(pinned)},
                    timeout=10.0
                )
                resp.raise_for_status()
                return True
            except Exception as exc:
                logger.error(f"Error pinning/unpinning chat_id={chat_id}: {exc}")
                return False

        # Dev fallback
        for u, arr in _IN_MEMORY_MEMORIES.items():
            for item in arr:
                if isinstance(item, dict) and item.get("id") == chat_id:
                    item["pinned"] = bool(pinned)
                    return True
        return False

    def rename_chat(self, chat_id: str, title: str) -> bool:
        """Renames a chat."""
        if self.is_configured:
            try:
                resp = httpx.patch(
                    f"{self.url}/rest/v1/chats?id=eq.{chat_id}",
                    headers=self._get_headers(),
                    json={"title": title},
                    timeout=10.0
                )
                resp.raise_for_status()
                return True
            except Exception as exc:
                logger.error(f"Error renaming chat_id={chat_id}: {exc}")
                return False

        for u, arr in _IN_MEMORY_MEMORIES.items():
            for item in arr:
                if isinstance(item, dict) and item.get("id") == chat_id:
                    item["title"] = title
                    return True
        return False
