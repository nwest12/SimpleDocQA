from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming question for the docs Q&A service."""

    question: str = Field(..., min_length=1, description="The user's question about Semantic Kernel")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the number of chunks to retrieve (defaults to service config)",
    )


class RetrievedChunk(BaseModel):
    """A single retrieved document chunk with its relevance score."""

    id: str
    content: str
    source: str | None = Field(default=None, description="Source file/URL the chunk came from")
    score: float = Field(..., description="Search relevance score (higher = more relevant)")


class QueryResponse(BaseModel):
    """Response containing retrieved chunks for a question."""

    question: str
    results: list[RetrievedChunk]
    count: int

class AskResponse(BaseModel):
    """Response containing a grounded answer plus the chunks it was based on."""

    question: str
    answer: str
    sources: list[RetrievedChunk]