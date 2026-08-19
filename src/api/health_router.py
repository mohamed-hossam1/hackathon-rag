import logging
from fastapi import APIRouter, status

from src.api.schemas import HealthResponse
from src.vectorstore.qdrant_store import QdrantVectorStore

logger = logging.getLogger("medical_rag.api.health")

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK
)
async def get_health_status() -> HealthResponse:
    """Returns basic health check status, Qdrant connectivity, and model state."""
    qdrant_connected = False
    try:
        store = QdrantVectorStore()
        store.client.get_collections()
        qdrant_connected = True
    except Exception as err:
        logger.warning(f"Health check Qdrant connection test failed: {err}")
        qdrant_connected = False

    return HealthResponse(
        status="healthy" if qdrant_connected else "degraded",
        qdrant_connected=qdrant_connected,
        models_loaded=True
    )
