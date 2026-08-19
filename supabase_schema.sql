-- Supabase SQL Schema for Medical RAG Dev Trace & AI-as-a-Judge Evaluation

-- 1. Table for logged Dev Queries
CREATE TABLE IF NOT EXISTS dev_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    abstained BOOLEAN DEFAULT FALSE,
    evidence_score FLOAT DEFAULT 0.0
);

-- 2. Table for Retrieval Traces (stores top 10 chunks per strategy)
CREATE TABLE IF NOT EXISTS retrieval_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES dev_queries(id) ON DELETE CASCADE,
    retrieval_method VARCHAR(50) NOT NULL, -- 'semantic', 'recursive', 'bm25', 'reranker'
    rank INT NOT NULL,                     -- 1 to 10
    chunk_id VARCHAR(255) NOT NULL,
    chunk_text TEXT NOT NULL,
    score FLOAT DEFAULT 0.0,
    filename VARCHAR(255),
    page_start INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Table for AI-as-a-Judge Evaluation Reports
CREATE TABLE IF NOT EXISTS evaluation_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES dev_queries(id) ON DELETE CASCADE,
    retrieval_method VARCHAR(50) NOT NULL,
    precision_at_3 FLOAT NOT NULL,         -- e.g. 0.67 (67%)
    precision_at_5 FLOAT NOT NULL,         -- e.g. 0.80 (80%)
    judgments JSONB NOT NULL,              -- list of {chunk_id, relevant, reason}
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indices for fast querying by query_id
CREATE INDEX IF NOT EXISTS idx_retrieval_traces_query_id ON retrieval_traces(query_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_reports_query_id ON evaluation_reports(query_id);

-- 4. Table for Uploaded Document Metadata & Ingestion Status
CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR(255) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    storage_path TEXT NOT NULL,
    status VARCHAR(50) NOT NULL,            -- 'queued', 'processing', 'completed', 'failed'
    total_pages INT DEFAULT NULL,
    error_message TEXT DEFAULT NULL,
    upload_timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Table for Document Chunks (for persistent BM25 re-hydration and retrieval)
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    document_id VARCHAR(255) REFERENCES documents(document_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    page_start INT NOT NULL,
    page_end INT NOT NULL,
    chunk_index INT NOT NULL,
    chunker_type VARCHAR(50) NOT NULL,      -- 'semantic', 'recursive'
    method VARCHAR(50) DEFAULT NULL,
    start_char INT NOT NULL,
    end_char INT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_chunker_type ON document_chunks(chunker_type);

