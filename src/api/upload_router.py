import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.api.schemas import (
    DocumentItem,
    DocumentListResponse,
    DocumentStatusResponse,
    ErrorResponse,
    UploadResponse,
)
from src.config import get_config
from src.models.document import Document, DocumentStatus, FileType
from src.queue.task_queue import TaskQueue
from src.services.document_store import DocumentStore
from src.services.ingestion_service import DocumentIngestionService

logger = logging.getLogger("medical_rag.api.upload")

router = APIRouter(tags=["Document Upload"])

ALLOWED_EXTENSIONS: Dict[str, FileType] = {
    "pdf": FileType.PDF,
    "docx": FileType.DOCX,
    "doc": FileType.DOC,
    "md": FileType.MD,
    "txt": FileType.TXT,
}

_task_queue_instance: Optional[TaskQueue] = None


def get_document_store() -> DocumentStore:
    return DocumentStore()


def get_ingestion_service() -> DocumentIngestionService:
    return DocumentIngestionService()


def get_task_queue() -> TaskQueue:
    global _task_queue_instance
    if _task_queue_instance is None:
        _task_queue_instance = TaskQueue()
        _task_queue_instance.start()
    return _task_queue_instance


def save_uploaded_file(file_content: bytes, filename: str, document_id: str) -> str:
    """Stores uploaded file to local filesystem fallback (or S3 if configured)."""
    config = get_config()
    storage_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(storage_dir, exist_ok=True)

    safe_filename = f"{document_id}_{Path(filename).name}"
    local_path = os.path.join(storage_dir, safe_filename)

    # Attempt S3 upload if S3 credentials/bucket are active
    s3_bucket = config.S3_BUCKET
    if s3_bucket and config.AWS_ACCESS_KEY_ID:
        try:
            import boto3
            s3_client = boto3.client(
                "s3",
                region_name=config.AWS_REGION,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY or None,
            )
            s3_key = f"documents/{safe_filename}"
            s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=file_content)
            logger.info(f"Uploaded '{filename}' to S3 bucket '{s3_bucket}' with key '{s3_key}'")
        except Exception as s3_err:
            logger.warning(f"S3 upload failed ({s3_err}), falling back to local storage path")

    # Save to local filesystem as target file
    with open(local_path, "wb") as f:
        f.write(file_content)

    logger.info(f"Saved uploaded file to '{local_path}'")
    return local_path


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request — unsupported file type or empty file"},
        413: {"model": ErrorResponse, "description": "Payload Too Large — file size exceeds limit"},
        500: {"model": ErrorResponse, "description": "Internal Server Error — storage failure"}
    }
)
async def upload_document(
    file: UploadFile = File(...),
    doc_store: DocumentStore = Depends(get_document_store),
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
    task_queue: TaskQueue = Depends(get_task_queue)
) -> UploadResponse:
    """Accepts document file upload (PDF, DOCX, DOC, MD, TXT), validates size/type, stores file, and queues background ingestion."""
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File appears to be empty or corrupted"
        )

    filename = file.filename
    ext = Path(filename).suffix.lstrip(".").lower()

    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected upload for filename '{filename}' with unsupported extension '.{ext}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '.{ext}'. Accepted: pdf, docx, doc, md, txt"
        )

    file_type = ALLOWED_EXTENSIONS[ext]
    config = get_config()
    max_mb = config.MAX_FILE_SIZE_MB
    max_bytes = max_mb * 1024 * 1024

    try:
        content = await file.read()
    except Exception as read_err:
        logger.error(f"Failed to read uploaded file '{filename}': {read_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File appears to be empty or corrupted"
        ) from read_err

    file_size = len(content)
    if file_size == 0:
        logger.warning(f"Uploaded file '{filename}' is 0 bytes")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File appears to be empty or corrupted"
        )

    if file_size > max_bytes:
        logger.warning(f"Uploaded file '{filename}' size ({file_size} bytes) exceeds limit ({max_bytes} bytes)")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum of {max_mb} MB"
        )

    document_id = str(uuid.uuid4())

    try:
        saved_path = save_uploaded_file(content, filename, document_id)
    except Exception as store_err:
        logger.error(f"Storage failure for document '{filename}': {store_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store document"
        ) from store_err

    # Create document record with status=QUEUED
    doc = Document(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        file_size_bytes=file_size,
        storage_path=saved_path,
        status=DocumentStatus.QUEUED
    )
    doc_store.add_document(doc)

    # Enqueue background processing task
    task_queue.enqueue_nowait(lambda: ingestion_service.ingest(document_id, saved_path))
    logger.info(f"Successfully enqueued processing task for document_id='{document_id}' ({filename})")

    return UploadResponse(
        document_id=document_id,
        filename=filename,
        file_type=ext,
        status=DocumentStatus.QUEUED,
        message="Document uploaded and queued for processing."
    )


@router.get(
    "/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Not Found — document ID does not exist"}
    }
)
async def get_document_status(
    document_id: str,
    doc_store: DocumentStore = Depends(get_document_store)
) -> DocumentStatusResponse:
    """Retrieves processing status and details for a specific uploaded document."""
    doc = doc_store.get_document(document_id)
    if not doc:
        logger.warning(f"Status requested for non-existent document_id='{document_id}'")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    return DocumentStatusResponse(
        document_id=doc.document_id,
        filename=doc.filename,
        status=doc.status,
        total_pages=doc.total_pages,
        error_message=doc.error_message
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK
)
async def list_documents(
    doc_store: DocumentStore = Depends(get_document_store)
) -> DocumentListResponse:
    """Lists all uploaded documents and their processing status."""
    docs = doc_store.list_documents()
    items = [
        DocumentItem(
            document_id=doc.document_id,
            filename=doc.filename,
            file_type=doc.file_type.value if hasattr(doc.file_type, "value") else str(doc.file_type),
            status=doc.status,
            upload_timestamp=doc.upload_timestamp,
            total_pages=doc.total_pages
        )
        for doc in docs
    ]

    return DocumentListResponse(documents=items)
