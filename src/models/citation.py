from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A reference linking a claim in the answer to a source chunk."""
    claim: str = Field(..., description="The specific claim text from the generated answer")
    chunk_id: str = Field(..., description="ID of the referenced source chunk")
    document_id: str = Field(..., description="Source document UUID")
    filename: str = Field(..., description="Source document filename")
    page_start: int = Field(..., ge=1, description="Start page of the cited chunk")
    page_end: int = Field(..., ge=1, description="End page of the cited chunk")


class CitationValidationResult(BaseModel):
    """The outcome of verifying whether a citation's evidence supports its claim."""
    claim: str = Field(..., description="The claim being validated")
    chunk_id: str = Field(..., description="ID of the cited chunk")
    chunk_text: str = Field(..., description="Text content of the cited chunk for auditability")
    supported: bool = Field(..., description="Whether the LLM validator determined the chunk supports the claim")
    reason: str = Field(..., description="Explanation of the validation judgment")
