# 🩺 Medical RAG System — Clinical Decision Support API

[![FastAPI](https://img.shields.io/badge/FastAPI-00558C?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-D12E66?style=for-the-badge)](https://qdrant.tech/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%2F%20OpenAI-F34B21?style=for-the-badge)](https://groq.com/)
[![AWS S3](https://img.shields.io/badge/AWS-S3_Storage-569A31?style=for-the-badge&logo=amazon-s3&logoColor=white)](https://aws.amazon.com/s3/)
[![Supabase](https://img.shields.io/badge/Supabase-Trace_Logs-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com/)

A production-grade, full-stack **Clinical Decision Support Retrieval-Augmented Generation (RAG)** application built with **FastAPI**, **Qdrant**, **BGE Embeddings & Reranker**, and **OpenAI-compatible LLM endpoints (Groq/OpenRouter/OpenAI)**.

The system ingests official medical guidelines (such as WHO documentation), parses multi-format clinical files, performs hybrid multi-stage retrieval with sentence-level semantic and recursive chunking, filters candidates via deduplication and cross-encoder reranking, and generates strictly grounded medical answers with automated citation verification and evidence thresholding.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Environment Configuration](#-environment-configuration)
- [API Reference & Usage](#-api-reference--usage)
  - [1. Health Check](#1-health-check)
  - [2. Document Upload & Management](#2-document-upload--management)
  - [3. RAG Query Execution](#3-rag-query-execution)
  - [4. Streaming RAG Query](#4-streaming-rag-query)
  - [5. Developer Tracing & Logs](#5-developer-tracing--logs)
  - [6. AI-as-a-Judge Evaluation](#6-ai-as-a-judge-evaluation)
- [Deployment Guide](#-deployment-guide)
  - [AWS EC2 Deployment (CI/CD)](#aws-ec2-deployment-cicd)
  - [AWS Lambda Deployment](#aws-lambda-deployment)
- [Medical Safety Disclaimer](#-medical-safety-disclaimer)

---

## ✨ Key Features

### 📄 Multi-Format Ingestion & OCR Processing
- Supports `.pdf`, `.docx`, `.doc`, `.md`, and `.txt` medical document files.
- Integrated **Tesseract OCR** via `pytesseract` and `Pillow` to recover text from scanned pages or image-only PDFs.
- Clean text extraction with header/footer removal and symbol normalization (`DocumentCleaner`).
- Non-blocking background ingestion task queue (`TaskQueue`) with document status tracking (`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`).
- Optional cloud sync with **AWS S3** for persistent storage.

### 🧠 Dual-Chunking Strategies
- **Semantic Chunker**: Sentence-level boundary splitting using `BAAI/bge-small-en-v1.5` embeddings to group semantically cohesive passages.
- **Recursive Chunker**: Sliding window character chunking with configurable size (default `512` tokens) and overlap (default `64` tokens) via `langchain-text-splitters`.
- Tracks precise metadata per chunk: document ID, filename, page numbers, and exact character offsets (`start_char`, `end_char`).

### 🔍 Hybrid Multi-Stage Retrieval
- **Semantic Retriever**: Dense vector similarity search across semantic chunk indices in Qdrant.
- **Recursive Retriever**: Vector search across recursive chunk indices in Qdrant.
- **BM25 Keyword Retriever**: Sparse keyword retrieval across all ingested document chunks using `rank-bm25`.

### ⚡ Candidate Deduplication & Reranking
- **Chunk Deduplicator**: Identifies and removes overlapping/redundant candidate passages prior to LLM context assembly.
- **BGE Cross-Encoder Reranker**: Reranks top candidates using `BAAI/bge-reranker-base` to optimize context relevance for generation.

### 🛡️ Grounded Generation, Citation Validation & Abstention
- Generates answers strictly grounded in retrieved evidence using OpenAI-compatible LLM backends (Groq, OpenRouter, OpenAI, vLLM, LM Studio, Ollama).
- **Citation Validator**: Validates AI claims against source chunks post-generation to eliminate hallucinations.
- **Evidence Threshold Guard**: Abstains from answering (`abstained=true`) when top evidence scores fall below threshold (`EVIDENCE_THRESHOLD=0.4`), returning a safe insufficient-evidence notice.
- Mandatory medical disclaimer appended to every answer response.

### 📊 Developer Tracing & Evaluation Framework
- **Dev Mode (`dev=true`)**: Exposes step-by-step intermediate state (`dev_trace`) revealing per-engine retrieved passages, deduplicated pools, cross-encoder scores, and citation validity.
- **Supabase Persistence**: Logs query execution traces and evaluation metrics for monitoring and analysis.
- **AI-as-a-Judge Evaluator**: Automated offline/online precision benchmarking (`Precision@3`, `Precision@5`) across semantic, recursive, BM25, and reranked retrieval pipelines using LLM judgment.

---

## 🏗️ System Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 User / Client Application              │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       │                                             │
               [ POST /upload ]                              [ POST /rag ]
                       │                                             │
                       ▼                                             ▼
       ┌───────────────────────────────┐             ┌───────────────────────────────┐
       │   Ingestion Queue & Storage   │             │   Hybrid Retrieval Pipeline   │
       └───────────────┬───────────────┘             └───────────────┬───────────────┘
                       │                                             │
         ┌─────────────┴─────────────┐                ┌──────────────┼──────────────┐
         ▼                           ▼                ▼              ▼              ▼
  ┌──────────────┐            ┌──────────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │ Format Parser│            │ AWS S3 Bucket│  │ Semantic  │  │ Recursive │  │   BM25    │
  │  & OCR Engine│            │ (Doc Backup) │  │ Retriever │  │ Retriever │  │ Retriever │
  └──────┬───────┘            └──────────────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
         │                                            │              │              │
         ▼                                            └──────────────┼──────────────┘
  ┌──────────────┐                                                   │
  │ Dual Chunker │                                                   ▼
  │ (Semantic +  │                                     ┌───────────────────────────┐
  │  Recursive)  │                                     │    Chunk Deduplicator     │
  └──────┬───────┘                                     └─────────────┬─────────────┘
         │                                                           │
         ▼                                                           ▼
  ┌──────────────┐                                     ┌───────────────────────────┐
  │ Vector store │                                     │   BGE Cross-Reranker      │
  │  (Qdrant)    │                                     └─────────────┬─────────────┘
  └──────────────┘                                                   │
                                                                     ▼
                                                       ┌───────────────────────────┐
                                                       │  OpenAI / Groq LLM Engine │
                                                       └─────────────┬─────────────┘
                                                                     │
                                                                     ▼
                                                       ┌───────────────────────────┐
                                                       │    Citation Validator     │
                                                       │  & Evidence Thresholding  │
                                                       └─────────────┬─────────────┘
                                                                     │
                                                                     ▼
                                                       ┌───────────────────────────┐
                                                       │   Grounded RAG Response   │
                                                       │  (Answer + Citations +    │
                                                       │   Dev Trace & Logs)       │
                                                       └───────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── main.py                          # FastAPI Application entry point & service wiring
├── requirements.txt                 # Python dependency manifest
├── .env.example                     # Environment variables configuration template
├── .github/workflows/
│   └── deploy.yml                   # CI/CD deployment workflow for AWS EC2
├── src/
│   ├── api/                         # FastAPI Routers & Pydantic Request/Response Schemas
│   │   ├── rag_router.py            # POST /rag and POST /rag/stream endpoints
│   │   ├── upload_router.py         # POST /upload, GET /documents, GET /documents/{id}/status
│   │   ├── dev_router.py            # Developer trace queries & strategy evaluation
│   │   ├── eval_router.py           # POST /evaluate AI-as-a-Judge benchmarking
│   │   ├── health_router.py         # GET /health check endpoint
│   │   └── schemas.py               # Data models for requests, responses, and errors
│   ├── chunking/                    # Chunker implementations
│   │   ├── semantic_chunker.py      # BGE Sentence embedding distance chunking
│   │   └── recursive_chunker.py     # Token sliding-window recursive chunking
│   ├── cleaning/                    # Text extraction & OCR normalization
│   │   ├── document_cleaner.py      # Noise reduction and format cleanup
│   │   └── ocr_processor.py         # PyTesseract fallback OCR pipeline
│   ├── db/                          # Database connections
│   │   └── supabase_service.py      # Supabase trace logging and query storage
│   ├── deduplication/               # Candidate overlap deduplication
│   │   └── chunk_deduplicator.py    # Overlap calculation & filtering
│   ├── embedding/                   # Embedding model wrapper
│   │   └── embedding_service.py     # FastEmbed / Sentence-Transformers (bge-small-en-v1.5)
│   ├── evaluation/                  # RAG benchmark evaluators
│   │   ├── llm_judge.py             # LLM Judge relevance evaluator
│   │   └── rag_evaluator.py         # Precision@K evaluator across retrieval strategies
│   ├── llm/                         # LLM service wrapper
│   │   └── llm_service.py           # OpenAI-compatible API interface (Groq / OpenRouter)
│   ├── models/                      # Core domain data classes & enums
│   │   ├── chunk.py                 # Chunk entity definition
│   │   ├── document.py              # Document status & file types
│   │   ├── retrieval.py             # Retrieval & Rerank result containers
│   │   └── response.py              # RAG final response & dev trace schemas
│   ├── parsers/                     # Multi-format document parsers
│   │   ├── pdf_parser.py            # PyMuPDF parser
│   │   ├── docx_parser.py           # python-docx parser
│   │   ├── doc_parser.py            # Legacy antiword parser
│   │   ├── markdown_parser.py       # Markdown parser
│   │   └── txt_parser.py            # Text file parser
│   ├── queue/                       # Async task queue
│   │   └── task_queue.py            # Threading background worker queue
│   ├── reranking/                   # Cross-encoder reranking
│   │   └── bge_reranker.py          # BAEI/bge-reranker-base model integration
│   ├── retrieval/                   # Hybrid retrieval implementations
│   │   ├── semantic_retriever.py    # Qdrant semantic vector search
│   │   ├── recursive_retriever.py   # Qdrant recursive vector search
│   │   └── bm25_retriever.py        # Sparse BM25 keyword index retriever
│   ├── services/                    # Orchestration & business logic
│   │   ├── rag_service.py           # RAG execution orchestrator
│   │   ├── ingestion_service.py     # Document processing pipeline
│   │   └── document_store.py        # In-memory & Supabase document index state
│   ├── validation/                  # Safety & citation verification
│   │   └── citation_validator.py    # Claim validation against source chunks
│   └── vectorstore/                 # Vector database client
│       └── qdrant_store.py          # Qdrant client wrapper & payload indexing
└── specs/                           # Specifications & design documentation
```

---

## ⚙️ Prerequisites

1. **Python 3.11 or higher**
2. **System Dependencies**:
   - **Tesseract OCR** (Required for scanned document text extraction):
     ```bash
     sudo apt-get update && sudo apt-get install -y tesseract-ocr libtesseract-dev
     ```
   - **antiword** (Required for legacy `.doc` parsing):
     ```bash
     sudo apt-get install -y antiword
     ```
3. **Qdrant Vector Database**:
   - Cloud cluster: Provision a free cluster at [Qdrant Cloud](https://cloud.qdrant.io/), or
   - Local Docker container: `docker run -p 6333:6333 qdrant/qdrant`
4. **LLM API Provider Key**:
   - [Groq Console](https://console.groq.com/) key (recommended for fast inference), or OpenRouter / OpenAI API key.

---

## 🚀 Quick Start

### 1. Clone the Repository & Activate Virtual Environment

```bash
cd /home/mohamed/Pictures/RAG/hackathon-rag
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Copy the example configuration and populate your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
LLM_API_KEY=gsk_your_groq_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b

QDRANT_URL=https://your-cluster-url.qdrant.tech:6333
QDRANT_API_KEY=your_qdrant_api_key_here
```

### 4. Launch the Server

Run the development server using FastAPI CLI or Uvicorn:

```bash
fastapi dev main.py
```
*or*
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access Interactive API Documentation

Open your browser to:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🛠️ Environment Configuration

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `LLM_API_KEY` | *(Required)* | API Key for OpenAI-compatible provider (Groq / OpenRouter / OpenAI) |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Base URL for LLM API service |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Target model identifier |
| `QDRANT_URL` | *(Required)* | URL for Qdrant vector database (Cloud or Local) |
| `QDRANT_API_KEY` | `""` | Qdrant Cloud access token |
| `COLLECTION_NAME` | `medical_documents` | Qdrant vector collection name |
| `S3_BUCKET` | `medical-rag-documents` | AWS S3 Bucket name for document uploads |
| `AWS_REGION` | `eu-central-1` | AWS region for S3 client |
| `AWS_ACCESS_KEY_ID` | `""` | AWS credentials ID |
| `AWS_SECRET_ACCESS_KEY` | `""` | AWS credentials secret |
| `SUPABASE_URL` | `""` | Supabase project URL for dev trace persistence |
| `SUPABASE_KEY` | `""` | Supabase API key |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed / Sentence-Transformers embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder model for reranking |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload file size limit (MB) |
| `SEMANTIC_CHUNK_THRESHOLD` | `0.50` | Cosine similarity threshold for semantic boundary splitting |
| `RECURSIVE_CHUNK_SIZE` | `1500` | Token/Character window size for recursive chunker |
| `RECURSIVE_CHUNK_OVERLAP` | `300` | Character overlap for recursive chunker |
| `RETRIEVAL_TOP_K` | `10` | Top-K initial passages retrieved per strategy |
| `RERANKER_TOP_K` | `5` | Top-K final passages passed to LLM context |
| `DEDUP_OVERLAP_THRESHOLD` | `0.80` | Overlap ratio threshold for chunk deduplication |
| `EVIDENCE_THRESHOLD` | `0.40` | Minimum score threshold required to prevent abstention |

---

## 📡 API Reference & Usage

### 1. Health Check
Verify system status and Qdrant connectivity.

```bash
curl -X GET http://127.0.0.1:8000/health
```

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "models_loaded": true
}
```

---

### 2. Document Upload & Management

#### Upload a Document
Accepts `.pdf`, `.docx`, `.doc`, `.md`, `.txt` files and queues them for asynchronous parsing, chunking, embedding, and vector indexing.

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@WHO-Guideline-Hypertension.pdf"
```

**Response (`202 Accepted`)**:
```json
{
  "document_id": "4b684820-9428-482a-9e1f-7b75217417e2",
  "filename": "WHO-Guideline-Hypertension.pdf",
  "file_type": "pdf",
  "status": "queued",
  "message": "Document uploaded and queued for processing."
}
```

#### Poll Processing Status

```bash
curl -X GET http://127.0.0.1:8000/documents/4b684820-9428-482a-9e1f-7b75217417e2/status
```

**Response (`200 OK`)**:
```json
{
  "document_id": "4b684820-9428-482a-9e1f-7b75217417e2",
  "filename": "WHO-Guideline-Hypertension.pdf",
  "status": "completed",
  "total_pages": 48,
  "error_message": null
}
```

#### List Ingested Documents

```bash
curl -X GET http://127.0.0.1:8000/documents
```

---

### 3. RAG Query Execution

#### Standard Query (Production Mode)

```bash
curl -X POST http://127.0.0.1:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the first-line pharmacological treatments for hypertension according to WHO?",
    "dev": false
  }'
```

**Response (`200 OK`)**:
```json
{
  "answer": "According to the WHO guidelines, initial pharmacological treatment for hypertension should include thiazide-like agents, ACE inhibitors (ACEis), angiotensin receptor blockers (ARBs), or long-acting dihydropyridine calcium channel blockers (CCBs).",
  "citations": [
    {
      "citation_id": "cit_1",
      "claim": "Initial treatment includes thiazide-like agents or ACEis/ARBs/CCBs",
      "chunk_id": "4b684820_semantic_14",
      "document_id": "4b684820-9428-482a-9e1f-7b75217417e2",
      "filename": "WHO-Guideline-Hypertension.pdf",
      "pages": [12],
      "reason": "Directly supported by recommendations on page 12."
    }
  ],
  "evidence_score": 0.842,
  "confidence_label": "high",
  "abstained": false,
  "disclaimer": "This answer is derived from ingested medical literature for decision-support purposes only and does not constitute formal medical diagnosis or advice.",
  "dev_trace": null
}
```

#### Query with Dev Trace (`dev=true`)

Enabling `dev=true` populates `dev_trace` with intermediate strategy outputs (semantic, recursive, BM25, deduplication, and reranker scores).

```bash
curl -X POST http://127.0.0.1:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the recommended threshold for initiating antihypertensive treatment?",
    "dev": true
  }'
```

---

### 4. Streaming RAG Query

Streams tokens in real-time via Server-Sent Events (SSE).

```bash
curl -N -X POST http://127.0.0.1:8000/rag/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Describe the dosage recommendations for ACE inhibitors.",
    "dev": false
  }'
```

---

### 5. Developer Tracing & Logs

- **List Logged Dev Queries**: `GET /dev/queries`
- **Get Full Trace for Query**: `GET /dev/queries/{query_id}/trace`
- **Run AI-as-a-Judge Evaluation on Query**: `POST /dev/queries/{query_id}/evaluate`

```bash
curl -X POST http://127.0.0.1:8000/dev/queries/b81f18a2-23c0-4f91-a12d-d01248aef019/evaluate
```

---

### 6. AI-as-a-Judge Evaluation

Benchmark retrieval relevance across all strategies (`semantic`, `recursive`, `bm25`, `reranker`) using automated batch LLM judgment.

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What are the diagnostic criteria for stage 2 hypertension?",
      "When should combination therapy be initiated?"
    ],
    "top_k": 5
  }'
```

---

## 🚢 Deployment Guide

### AWS EC2 Deployment (CI/CD)

The repository includes an automated GitHub Actions deployment workflow at `.github/workflows/deploy.yml`.

#### Setup Steps:
1. Provision an **Ubuntu 22.04 LTS EC2 Instance**.
2. Configure GitHub Repository Secrets:
   - `EC2_HOST`: EC2 public IP or DNS
   - `EC2_SSH_KEY`: Private SSH key for Ubuntu user
3. Set up Systemd service on EC2 (`/etc/systemd/system/rag.service`):

```ini
[Unit]
Description=Medical RAG FastAPI Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/rag-app
ExecStart=/home/ubuntu/rag-app/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
```

4. Every push to the `main` branch automatically triggers code update, dependency installation, and service restart via SSH.

---

### AWS Lambda Deployment

The application includes a `Mangum` handler in `main.py` allowing serverless execution on AWS Lambda + API Gateway:

```python
from mangum import Mangum
from main import app

handler = Mangum(app)
```

Packaging for AWS Lambda can be performed using AWS SAM, Serverless Framework, or Docker container images deployed to AWS ECR.

---

## ⚕️ Medical Safety Disclaimer

> **IMPORTANT**: This application is built strictly as a decision-support demonstration tool for retrieving information from validated clinical guidelines. It is **not** intended to diagnose conditions, prescribe medications, or replace professional clinical judgment. Always consult a licensed medical professional for patient care.