from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Status lifecycle enum for uploaded documents."""
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(str, Enum):
    """Supported document file formats."""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    MD = "md"
    TXT = "txt"


class ExtractionMethod(str, Enum):
    """Method used for text extraction."""
    NATIVE = "native"
    OCR = "ocr"


class TableData(BaseModel):
    """Represents a single table extracted from a document page."""
    table_index: int = Field(..., ge=0, description="0-indexed table position on page")
    headers: Optional[List[str]] = Field(default=None, description="Extracted column headers if detected")
    rows: List[List[str]] = Field(..., description="2D list of row cell text values")
    caption: Optional[str] = Field(default=None, description="Table caption or surrounding context")
    normalized_text: str = Field(..., description="Pipe-delimited text representation of table")


class DocumentPage(BaseModel):
    """Represents a single page extracted from a document during parsing."""
    document_id: str = Field(..., description="Parent Document UUID")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    raw_text: str = Field(..., description="Raw text extracted before cleaning")
    cleaned_text: str = Field(..., description="Text after cleaning pipeline execution")
    extraction_method: ExtractionMethod = Field(..., description="Extraction method used: native or ocr")
    tables: List[TableData] = Field(default_factory=list, description="List of tables extracted from page")
    has_meaningful_text: bool = Field(..., description="Whether native text extraction produced usable text")


class Document(BaseModel):
    """Represents an uploaded medical document entity."""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique document UUID")
    filename: str = Field(..., description="Original uploaded filename")
    file_type: FileType = Field(..., description="Detected file format")
    file_size_bytes: int = Field(..., gt=0, description="File size in bytes")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC upload timestamp")
    storage_path: str = Field(..., description="Storage key or path (e.g., S3 object key)")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED, description="Current processing state")
    error_message: Optional[str] = Field(default=None, description="Failure detail if status is failed")
    total_pages: Optional[int] = Field(default=None, ge=0, description="Total document pages after parsing")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary document metadata")
