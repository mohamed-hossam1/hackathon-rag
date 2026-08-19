import logging
from typing import List, Optional
from fastembed import TextEmbedding

from src.config import get_config

logger = logging.getLogger("medical_rag.embedding")


class EmbeddingService:
    """Service for generating 384-dimensional dense vector embeddings using FastEmbed TextEmbedding."""

    def __init__(self, model_name: Optional[str] = None):
        config = get_config()
        self.model_name = model_name or config.EMBEDDING_MODEL
        logger.info(f"Initializing FastEmbed TextEmbedding model: {self.model_name}")
        self._model = TextEmbedding(model_name=self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of text strings into 384-dimensional dense vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vector lists (each of length 384).
        """
        if not texts:
            return []

        embeddings_generator = self._model.embed(texts)
        return [e.tolist() for e in embeddings_generator]

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single search query into a 384-dimensional vector.

        Args:
            query: Query string.

        Returns:
            Vector of floats (length 384).
        """
        results = self.embed_texts([query])
        return results[0] if results else []
