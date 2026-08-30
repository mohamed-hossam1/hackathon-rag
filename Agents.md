# Backend Agents

This file maps the backend "agents" (service classes, retrievers, storage adapters, API routers) to their responsibilities and key implementation files so engineers can quickly find where behavior lives.

## Overview
- Language & framework: Python + FastAPI
- Entry point & lifecycle: `main.py` (defines `lifespan` — initializes singletons, starts `TaskQueue`, mounts routers, and exports a Mangum `handler` for AWS Lambda)
- Key responsibilities: ingestion, chunking, embedding, vector storage, retrieval, reranking, LLM generation, citation validation, dev tracing, and evaluation.

## API Routers (top-level)
- `src/api/rag_router.py` — `/rag` and `/rag/stream` endpoints (synchronous and SSE streaming). See `query_rag` and `query_rag_stream` handlers.
- `src/api/upload_router.py` — document upload and ingestion endpoints; wires `DocumentIngestionService` and `DocumentStore` dependencies.
- `src/api/auth_router.py` — authentication endpoints and user/session helpers.
- `src/api/eval_router.py` — evaluation endpoints for LLM/judge flows.
- `src/api/health_router.py` — health checks for components (vector DB, etc.).

## Core Service Agents (modules)
- Document ingestion: `src/services/ingestion_service.py` (DocumentIngestionService)
  - Pipeline: parser → OCR fallback → cleaning → dual chunking (semantic + recursive) → embedding → upsert to Qdrant → BM25 index rebuild.
- Chunking: `src/chunking/semantic_chunker.py`, `src/chunking/recursive_chunker.py`
- Cleaning/OCR: `src/cleaning/document_cleaner.py`, `src/cleaning/ocr_processor.py`
- Embedding: `src/embedding/embedding_service.py` (uses FastEmbed TextEmbedding)
- Vector store: `src/vectorstore/qdrant_store.py` (Qdrant client wrapper — create/search/upsert/delete/health_check)
- Retrievals: `src/retrieval/semantic_retriever.py`, `src/retrieval/recursive_retriever.py`, `src/retrieval/bm25_retriever.py`
- Reranking: `src/reranking/bge_reranker.py` (BGE reranker used to promote top candidates)
- Deduplication: `src/deduplication/chunk_deduplicator.py` (remove duplicate chunks across retrievers)

## RAG Orchestrator
- `src/services/rag_service.py` — core orchestrator implementing the tri-hybrid retrieval pattern:
  1. Parallel retrieval (semantic, recursive, BM25)
  2. Normalize & deduplicate candidate chunks
  3. Rerank with BGE reranker and select top-K
  4. Assemble context prompt and call LLM (`src/llm/llm_service.py`)
  5. Parse citations, validate with `src/validation/citation_validator.py`, compute evidence score and confidence label
  6. Optionally persist dev traces to Supabase (background task)

## LLM & Generation
- `src/llm/llm_service.py` (OpenAI-compatible wrapper) — streaming and non-streaming generation helpers, configurable by `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` in config.

## Storage & Persistence
- Document metadata & saved memories: `src/services/document_store.py` and `src/db/supabase_service.py` (Supabase interactions and dev-trace persistence).
- Vector storage: `src/vectorstore/qdrant_store.py` (Qdrant collection management and payload indexing).

## Background & Utilities
- `src/queue/task_queue.py` — background worker for ingestion tasks and asynchronous jobs.
- `src/evaluation/*` and `src/validation/*` — LLM judge, RAG evaluators, and citation validation helpers.

## Configuration / Environment
Key configuration values used in the code (exposed via `src.config.get_config()`):
- Vector DB: `QDRANT_URL`, `QDRANT_API_KEY`, `COLLECTION_NAME`, `QDRANT_TIMEOUT`, `QDRANT_BATCH_SIZE`
- Embeddings: `EMBEDDING_MODEL`
- LLM: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`
- Retrieval & scoring: `RETRIEVAL_TOP_K`, `RERANKER_TOP_K`, `EVIDENCE_THRESHOLD`
- Supabase: URL / keys used by `src/db/supabase_service.py`

## How the app starts
- `main.py` initializes singletons (embedding service, vector store, document store, chunkers, retrievers, reranker, LLM, ingestion service, RAG service) during FastAPI `lifespan` and registers dependency overrides so route handlers reuse those instances.

## Key files to inspect
- [main.py](main.py#L1) — app entry and lifecycle
- [src/services/rag_service.py](src/services/rag_service.py#L1) — RAG orchestration
- [src/services/ingestion_service.py](src/services/ingestion_service.py#L1) — ingestion pipeline
- [src/vectorstore/qdrant_store.py](src/vectorstore/qdrant_store.py#L1) — vector DB adapter
- [src/llm/llm_service.py](src/llm/llm_service.py#L1) — LLM provider wrapper

## Recommendations / Next steps
- Add a short `docs/ARCHITECTURE.md` or `backend/README.md` that diagrams the tri-hybrid retrieval → rerank → LLM flow and lists required env variables.
- Add health-check endpoints that surface embedding model, Qdrant status, and LLM connectivity for operational monitoring.


