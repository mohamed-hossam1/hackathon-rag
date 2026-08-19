import logging
from threading import Lock
from typing import Dict, List, Optional
from datetime import datetime

from src.db.supabase_service import SupabaseService
from src.models.chunk import Chunk, ChunkerType
from src.models.document import Document, DocumentStatus, FileType

logger = logging.getLogger("medical_rag.services.document_store")


class DocumentStore:
    """Thread-safe persistent & in-memory store for tracking uploaded documents and their extracted chunks."""

    _instance: Optional["DocumentStore"] = None
    _lock = Lock()

    _documents: Dict[str, Document]
    _document_chunks: Dict[str, List[Chunk]]
    _store_lock: Lock
    _supabase: SupabaseService

    def __new__(cls) -> "DocumentStore":
        """Singleton pattern for central DocumentStore state across application routes."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._documents = {}
                instance._document_chunks = {}
                instance._store_lock = Lock()
                instance._supabase = SupabaseService()
                cls._instance = instance
                logger.info("Initialized DocumentStore singleton instance")
            return cls._instance

    def load_from_supabase(self) -> None:
        """Loads all document metadata and chunks from Supabase to re-hydrate in-memory state on startup."""
        if not self._supabase.is_configured:
            logger.info("Supabase not configured; skipping startup state re-hydration")
            return

        with self._store_lock:
            try:
                raw_docs = self._supabase.get_all_documents()
                raw_chunks = self._supabase.get_all_chunks()

                loaded_doc_count = 0
                for d in raw_docs:
                    try:
                        doc_id = d["document_id"]
                        ft_val = d.get("file_type", "txt")
                        try:
                            file_type_enum = FileType(ft_val)
                        except ValueError:
                            file_type_enum = FileType.TXT

                        status_val = d.get("status", "queued")
                        try:
                            status_enum = DocumentStatus(status_val)
                        except ValueError:
                            status_enum = DocumentStatus.QUEUED

                        ts_str = d.get("upload_timestamp")
                        upload_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.now()

                        doc = Document(
                            document_id=doc_id,
                            filename=d.get("filename", "unknown"),
                            file_type=file_type_enum,
                            file_size_bytes=int(d.get("file_size_bytes", 0)),
                            storage_path=d.get("storage_path", ""),
                            status=status_enum,
                            total_pages=d.get("total_pages"),
                            error_message=d.get("error_message"),
                            upload_timestamp=upload_ts
                        )
                        self._documents[doc_id] = doc
                        self._document_chunks[doc_id] = []
                        loaded_doc_count += 1
                    except Exception as doc_err:
                        logger.error(f"Error parsing document row from Supabase: {doc_err}")

                loaded_chunk_count = 0
                for c in raw_chunks:
                    try:
                        doc_id = c["document_id"]
                        chunker_type_val = c.get("chunker_type", "semantic")
                        try:
                            chunker_type_enum = ChunkerType(chunker_type_val)
                        except ValueError:
                            chunker_type_enum = ChunkerType.SEMANTIC

                        chunk = Chunk(
                            chunk_id=c["chunk_id"],
                            text=c["text"],
                            document_id=doc_id,
                            filename=c.get("filename", ""),
                            page_start=int(c.get("page_start", 1)),
                            page_end=int(c.get("page_end", 1)),
                            chunk_index=int(c.get("chunk_index", 0)),
                            chunker_type=chunker_type_enum,
                            method=str(c.get("method") or "native"),

                            start_char=int(c.get("start_char", 0)),
                            end_char=int(c.get("end_char", 0))
                        )
                        if doc_id in self._document_chunks:
                            self._document_chunks[doc_id].append(chunk)
                        else:
                            self._document_chunks[doc_id] = [chunk]
                        loaded_chunk_count += 1
                    except Exception as chunk_err:
                        logger.error(f"Error parsing chunk row from Supabase: {chunk_err}")

                logger.info(
                    f"Successfully re-hydrated DocumentStore from Supabase: "
                    f"{loaded_doc_count} documents, {loaded_chunk_count} chunks"
                )
            except Exception as err:
                logger.error(f"Failed to re-hydrate DocumentStore from Supabase: {err}")

    def add_document(self, document: Document) -> None:
        """Stores a new document entity.

        Args:
            document: Document model instance.
        """
        with self._store_lock:
            self._documents[document.document_id] = document
            self._document_chunks[document.document_id] = []
            logger.info(f"Added document_id='{document.document_id}' ({document.filename}) to DocumentStore")

        # Persist to Supabase asynchronously / background
        doc_dict = {
            "document_id": document.document_id,
            "filename": document.filename,
            "file_type": document.file_type.value if hasattr(document.file_type, "value") else str(document.file_type),
            "file_size_bytes": document.file_size_bytes,
            "storage_path": document.storage_path,
            "status": document.status.value if hasattr(document.status, "value") else str(document.status),
            "total_pages": document.total_pages,
            "error_message": document.error_message,
        }
        self._supabase.save_document(doc_dict)

    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieves a document entity by UUID.

        Args:
            document_id: Target document UUID string.

        Returns:
            Document object if found, else None.
        """
        with self._store_lock:
            return self._documents.get(document_id)

    def list_documents(self) -> List[Document]:
        """Lists all registered documents sorted by upload timestamp descending."""
        with self._store_lock:
            docs = list(self._documents.values())
            docs.sort(key=lambda d: d.upload_timestamp, reverse=True)
            return docs

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        total_pages: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> Optional[Document]:
        """Updates document status, total pages, and error message.

        Args:
            document_id: Target document UUID string.
            status: New DocumentStatus enum.
            total_pages: Optional total page count.
            error_message: Optional failure explanation string.

        Returns:
            Updated Document object if found, else None.
        """
        with self._store_lock:
            doc = self._documents.get(document_id)
            if not doc:
                logger.warning(f"Failed to update status: document_id='{document_id}' not found")
                return None

            doc.status = status
            if total_pages is not None:
                doc.total_pages = total_pages
            if error_message is not None:
                doc.error_message = error_message

            logger.info(f"Updated document_id='{document_id}' status to '{status.value}'")

        status_str = status.value if hasattr(status, "value") else str(status)
        self._supabase.update_document_status(
            document_id=document_id,
            status=status_str,
            total_pages=total_pages,
            error_message=error_message
        )
        return doc

    def add_chunks(self, document_id: str, chunks: List[Chunk]) -> None:
        """Associates extracted chunks with a document.

        Args:
            document_id: Target document UUID string.
            chunks: List of Chunk objects.
        """
        with self._store_lock:
            if document_id not in self._documents:
                logger.warning(f"Cannot add chunks: document_id='{document_id}' not found")
                return

            self._document_chunks[document_id] = chunks
            logger.info(f"Added {len(chunks)} chunks to document_id='{document_id}' in DocumentStore")

        chunks_dicts = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "filename": c.filename,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "chunk_index": c.chunk_index,
                "chunker_type": c.chunker_type.value if hasattr(c.chunker_type, "value") else str(c.chunker_type),
                "method": c.method,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in chunks
        ]
        self._supabase.save_chunks(chunks_dicts)

    def get_document_chunks(self, document_id: str) -> List[Chunk]:
        """Retrieves chunks associated with a specific document."""
        with self._store_lock:
            return self._document_chunks.get(document_id, [])

    def get_all_chunks(self) -> List[Chunk]:
        """Returns all chunks across all completed documents for BM25 index initialization."""
        with self._store_lock:
            all_chunks = []
            for chunks in self._document_chunks.values():
                all_chunks.extend(chunks)
            return all_chunks

    def clear(self) -> None:
        """Clears all in-memory document state (useful for testing)."""
        with self._store_lock:
            self._documents.clear()
            self._document_chunks.clear()
            logger.info("Cleared DocumentStore state")

