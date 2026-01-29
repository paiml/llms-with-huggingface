"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response.

    Example:
        {"status": "healthy"}
    """

    status: str = Field(
        default="healthy",
        description="Health status of the service",
        examples=["healthy", "unhealthy"],
    )


class ChatRequest(BaseModel):
    """Request body for chat endpoint.

    Example:
        {"message": "Hello, how are you?", "system_prompt": "You are helpful."}
    """

    message: str = Field(
        ...,
        description="The user's message to send to the LLM",
        min_length=1,
        examples=["What is machine learning?"],
    )
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt to set the assistant's behavior",
        examples=["You are a helpful assistant.", "You are a coding expert."],
    )


class ChatResponse(BaseModel):
    """Response body for chat endpoint.

    Example:
        {"response": "I'm doing well, thank you!"}
    """

    response: str = Field(
        ...,
        description="The assistant's response message",
        examples=["Machine learning is a subset of AI..."],
    )


class ResearchRequest(BaseModel):
    """Request body for research endpoint.

    Example:
        {"query": "What are the latest developments in AI?"}
    """

    query: str = Field(
        ...,
        description="The research query to investigate",
        min_length=1,
        examples=["What are the benefits of RAG systems?"],
    )


class ResearchResponse(BaseModel):
    """Response body for research endpoint.

    Example:
        {"answer": "Based on my research...", "sources": ["doc1", "doc2"]}
    """

    answer: str = Field(
        ...,
        description="The research answer from the agent",
        examples=["Based on the knowledge base and web search..."],
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of sources used in the research",
        examples=[["Internal KB: AI Overview", "Web: arxiv.org"]],
    )


class DocumentRequest(BaseModel):
    """Request body for adding documents to knowledge base.

    Example:
        {"title": "AI Overview", "content": "Artificial intelligence is..."}
    """

    title: str = Field(
        ...,
        description="Title of the document",
        min_length=1,
        examples=["Introduction to Machine Learning"],
    )
    content: str = Field(
        ...,
        description="Content of the document",
        min_length=1,
        examples=["Machine learning is a method of data analysis..."],
    )


class DocumentResponse(BaseModel):
    """Response body for document operations.

    Example:
        {"status": "success", "message": "Document added"}
    """

    status: str = Field(
        ...,
        description="Operation status",
        examples=["success", "error"],
    )
    message: str = Field(
        ...,
        description="Descriptive message about the operation",
        examples=["Document added to knowledge base"],
    )


class ErrorResponse(BaseModel):
    """Standard error response format.

    Example:
        {"detail": "Document not found", "error_code": "NOT_FOUND"}
    """

    detail: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Invalid request parameters"],
    )
    error_code: str = Field(
        default="INTERNAL_ERROR",
        description="Machine-readable error code",
        examples=["VALIDATION_ERROR", "NOT_FOUND", "INTERNAL_ERROR"],
    )
