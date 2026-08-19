import logging
from threading import Lock
from typing import Dict, List, Optional

from src.models.chunk import Chunk
from src.models.document import Document, DocumentStatus

logger = logging.getLogger("medical_rag.services.document_store")


class DocumentStore:
    """Thread-safe in-memory store for tracking uploaded documents and their extracted chunks."""

    _instance: Optional["DocumentStore"] = None
    _lock = Lock()

    _documents: Dict[str, Document]
    _document_chunks: Dict[str, List[Chunk]]
    _store_lock: Lock

    def __new__(cls) -> "DocumentStore":
        """Singleton pattern for central DocumentStore state across application routes."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._documents = {}
                instance._document_chunks = {}
                instance._store_lock = Lock()
                cls._instance = instance
                logger.info("Initialized DocumentStore singleton instance")
            return cls._instance

    def add_document(self, document: Document) -> None:
        """Stores a new document entity.

        Args:
            document: Document model instance.
        """
        with self._store_lock:
            self._documents[document.document_id] = document
            self._document_chunks[document.document_id] = []
            logger.info(f"Added document_id='{document.document_id}' ({document.filename}) to DocumentStore")

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
