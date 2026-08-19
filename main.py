import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.eval_router import router as eval_router
from src.api.health_router import router as health_router
from src.api.rag_router import router as rag_router, get_rag_service
from src.config import get_config
from src.deduplication.chunk_deduplicator import ChunkDeduplicator
from src.embedding.embedding_service import EmbeddingService
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.rag_evaluator import RAGEvaluator
from src.llm.llm_service import OpenAILikeLLMService
from src.reranking.bge_reranker import BGEReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.recursive_retriever import RecursiveRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.services.document_store import DocumentStore
from src.services.rag_service import RAGService
from src.validation.citation_validator import CitationValidator
from src.vectorstore.qdrant_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("medical_rag.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle event handler."""
    logger.info("Initializing Medical RAG Application components...")

    # Load configuration
    config = get_config()

    # Core Infrastructure Services
    embedding_service = EmbeddingService()
    vector_store = QdrantVectorStore()
    document_store = DocumentStore()

    # Retrievers
    semantic_retriever = SemanticRetriever(
        embedding_service=embedding_service, vector_store=vector_store
    )
    recursive_retriever = RecursiveRetriever(
        embedding_service=embedding_service, vector_store=vector_store
    )

    # Initialize BM25 index with any existing document chunks
    bm25_retriever = BM25Retriever(chunks=document_store.get_all_chunks())

    # Reranking & Deduplication
    deduplicator = ChunkDeduplicator()
    reranker = BGEReranker()

    # LLM & Validation
    llm_service = OpenAILikeLLMService()
    citation_validator = CitationValidator(llm_service=llm_service)

    # RAG Orchestrator Service
    rag_service = RAGService(
        semantic_retriever=semantic_retriever,
        recursive_retriever=recursive_retriever,
        bm25_retriever=bm25_retriever,
        deduplicator=deduplicator,
        reranker=reranker,
        llm_service=llm_service,
        citation_validator=citation_validator,
        evidence_threshold=config.EVIDENCE_THRESHOLD
    )

    # Override FastAPI dependency for singleton RAGService reuse
    app.dependency_overrides[get_rag_service] = lambda: rag_service

    logger.info("Medical RAG Application initialization complete!")
    yield
    logger.info("Shutting down Medical RAG Application...")


app = FastAPI(
    title="Medical RAG System API",
    description="Full-Stack Clinical Decision Support RAG Application",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(rag_router)
app.include_router(health_router)
app.include_router(eval_router)

# Mount Frontend Static Files if directory exists
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")