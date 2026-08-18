from enum import Enum
from pydantic import BaseModel, Field, model_validator


class ChunkerType(str, Enum):
    """Chunking strategy enum."""
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


class Chunk(BaseModel):
    """Represents a segment of document text prepared for embedding and retrieval."""

    chunk_id: str = Field(..., description="Deterministic ID: {document_id}_{chunker_type}_{chunk_index}")
    text: str = Field(..., min_length=1, description="Chunk text content")
    document_id: str = Field(..., description="Parent document UUID")
    filename: str = Field(..., description="Source document filename")
    page_start: int = Field(..., ge=1, description="First page this chunk covers (1-indexed)")
    page_end: int = Field(..., ge=1, description="Last page this chunk covers (1-indexed)")
    chunk_index: int = Field(..., ge=0, description="Sequential index within strategy output")
    chunker_type: ChunkerType = Field(..., description="Chunking strategy used: semantic or recursive")
    method: str = Field(..., description="Text extraction method of source document: native or ocr")
    start_char: int = Field(..., ge=0, description="Start character offset in normalized document text")
    end_char: int = Field(..., description="End character offset in normalized document text")

    @model_validator(mode="after")
    def validate_offsets_and_pages(self) -> "Chunk":
        if self.page_end < self.page_start:
            raise ValueError(f"page_end ({self.page_end}) must be >= page_start ({self.page_start})")
        if self.end_char <= self.start_char:
            raise ValueError(f"end_char ({self.end_char}) must be > start_char ({self.start_char})")
        return self
