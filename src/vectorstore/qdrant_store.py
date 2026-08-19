import logging
from typing import Any, Dict, List, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import get_config
from src.models.chunk import Chunk
from src.models.retrieval import RetrievalMethod, RetrievalResult
from src.vectorstore.base import VectorStore

logger = logging.getLogger("medical_rag.vectorstore")


class QdrantVectorStore(VectorStore):
    """Qdrant vector database implementation for dense vector retrieval."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_size: int = 384
    ):
        config = get_config()
        self.url = url or config.QDRANT_URL
        self.api_key = api_key or config.QDRANT_API_KEY
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.vector_size = vector_size

        if self.api_key and self.url:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        elif self.url:
            self.client = QdrantClient(url=self.url)
        else:
            self.client = QdrantClient(":memory:")

    def create_collection_if_not_exists(self) -> bool:
        """Ensures target Qdrant collection exists with cosine distance and payload indexes."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' (size={self.vector_size}, distance=COSINE)")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size,
                        distance=qmodels.Distance.COSINE
                    )
                )
                # Create payload indexes for efficient filtering
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="document_id",
                        field_schema=qmodels.PayloadSchemaType.KEYWORD
                    )
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="chunker_type",
                        field_schema=qmodels.PayloadSchemaType.KEYWORD
                    )
                except Exception as index_err:
                    logger.debug(f"Payload index notice: {index_err}")
            return True
        except Exception as err:
            logger.error(f"Failed to create/check Qdrant collection '{self.collection_name}': {err}")
            return False

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> bool:
        """Upserts chunks and their embedding vectors into Qdrant."""
        if not chunks or not vectors:
            return True

        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatched chunks count ({len(chunks)}) and vectors count ({len(vectors)})")

        self.create_collection_if_not_exists()

        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk.model_dump()
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully upserted {len(points)} chunks into '{self.collection_name}'")
            return True
        except Exception as err:
            logger.error(f"Failed to upsert chunks into Qdrant: {err}")
            raise err

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        chunker_type_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """Performs vector similarity search in Qdrant with optional chunker_type filter."""
        self.create_collection_if_not_exists()

        query_filter = None
        if chunker_type_filter:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="chunker_type",
                        match=qmodels.MatchValue(value=chunker_type_filter)
                    )
                ]
            )

        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=query_filter
                )
                hits = response.points
            elif hasattr(self.client, "search"):
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=query_filter
                )
            else:
                raise AttributeError("QdrantClient has neither query_points nor search method")

            results: List[RetrievalResult] = []
            for hit in hits:
                payload = hit.payload or {}
                chunk = Chunk(**payload)
                method_str = payload.get("chunker_type", "semantic")
                retrieval_method = RetrievalMethod(method_str)
                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=float(hit.score),
                        retrieval_method=retrieval_method
                    )
                )

            return results
        except Exception as err:
            logger.error(f"Qdrant vector search failed: {err}")
            return []

    def delete(self, document_id: str) -> bool:
        """Deletes all vector points associated with document_id."""
        self.create_collection_if_not_exists()

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id",
                                match=qmodels.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted vector points for document_id='{document_id}' from '{self.collection_name}'")
            return True
        except Exception as err:
            logger.error(f"Failed to delete document_id='{document_id}' from Qdrant: {err}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Checks connectivity to Qdrant cluster."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            return {
                "status": "healthy",
                "qdrant_connected": True,
                "collection_exists": exists,
                "collection_name": self.collection_name
            }
        except Exception as err:
            logger.error(f"Qdrant health check failed: {err}")
            return {
                "status": "unhealthy",
                "qdrant_connected": False,
                "error": str(err)
            }
