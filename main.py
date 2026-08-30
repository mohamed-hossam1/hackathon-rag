import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mangum import Mangum

from src.api.auth_router import router as auth_router
from src.api.dev_router import router as dev_router
from src.api.eval_router import router as eval_router
from src.api.health_router import router as health_router
from src.api.rag_router import router as rag_router, get_rag_service
from src.api.chat_router import router as chat_router
from src.api.upload_router import (
    router as upload_router,
    get_document_store,
    get_ingestion_service,
    get_task_queue,
)
from src.chunking.recursive_chunker import RecursiveChunker
from src.chunking.semantic_chunker import SemanticChunker
from src.cleaning.document_cleaner import DocumentCleaner
from src.cleaning.ocr_processor import OCRProcessor
from src.config import get_config
from src.deduplication.chunk_deduplicator import ChunkDeduplicator
from src.embedding.embedding_service import EmbeddingService
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.rag_evaluator import RAGEvaluator
from src.llm.llm_service import OpenAILikeLLMService
from src.models.document import FileType
from src.parsers.base import Parser
from src.parsers.doc_parser import DOCParser
from src.parsers.docx_parser import DOCXParser
from src.parsers.markdown_parser import MarkdownParser
from src.parsers.pdf_parser import PDFParser
from src.parsers.txt_parser import TXTParser
from src.queue.task_queue import TaskQueue
from src.reranking.bge_reranker import BGEReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.recursive_retriever import RecursiveRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.services.document_store import DocumentStore
from src.services.ingestion_service import DocumentIngestionService
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
    document_store.load_from_supabase()

    # Parsers
    parsers: dict[str, Parser] = {
        FileType.PDF.value: PDFParser(),
        FileType.DOCX.value: DOCXParser(),
        FileType.DOC.value: DOCParser(),
        FileType.MD.value: MarkdownParser(),
        FileType.TXT.value: TXTParser(),
    }

    # Cleaning & Chunking
    document_cleaner = DocumentCleaner()
    ocr_processor = OCRProcessor()
    semantic_chunker = SemanticChunker(embedding_service=embedding_service)
    recursive_chunker = RecursiveChunker()

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

    # Document Ingestion Service & Background Task Queue
    ingestion_service = DocumentIngestionService(
        document_store=document_store,
        embedding_service=embedding_service,
        qdrant_store=vector_store,
        bm25_retriever=bm25_retriever,
        semantic_chunker=semantic_chunker,
        recursive_chunker=recursive_chunker,
        document_cleaner=document_cleaner,
        ocr_processor=ocr_processor,
        parsers=parsers,
    )
    task_queue = TaskQueue()
    task_queue.start()

    # RAG Orchestrator Service
    rag_service = RAGService(
        semantic_retriever=semantic_retriever,
        recursive_retriever=recursive_retriever,
        bm25_retriever=bm25_retriever,
        deduplicator=deduplicator,
        reranker=reranker,
        llm_service=llm_service,
        citation_validator=citation_validator,
        evidence_threshold=config.EVIDENCE_THRESHOLD,
    )

    # Override FastAPI dependencies for singleton component reuse across API routes
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    app.dependency_overrides[get_document_store] = lambda: document_store
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[get_task_queue] = lambda: task_queue

    logger.info("Medical RAG Application initialization complete!")
    yield
    logger.info("Shutting down Medical RAG Application...")
    await task_queue.stop()


app = FastAPI(
    title="Medical RAG System API",
    description="Full-Stack Clinical Decision Support RAG Application",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "https://medical-rag-ui-pied.vercel.app"
    ],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(upload_router)
app.include_router(health_router)
app.include_router(eval_router)
app.include_router(dev_router)

# Mount Frontend Static Files if directory exists
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# AWS Lambda handler using Mangum adapter
handler = Mangum(app)