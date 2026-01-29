"""Research endpoint routes."""

import logging

from fastapi import APIRouter, HTTPException

from src.agents.research_agent import ResearchAgent
from src.models.schemas import (
    DocumentRequest,
    DocumentResponse,
    ErrorResponse,
    ResearchRequest,
    ResearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["research"])

# Lazy initialization of research agent
_research_agent: ResearchAgent | None = None


def get_research_agent() -> ResearchAgent:
    """Get or create the research agent singleton.

    Returns:
        ResearchAgent instance.
    """
    global _research_agent
    if _research_agent is None:
        _research_agent = ResearchAgent()
    return _research_agent


@router.post(
    "/research",
    response_model=ResearchResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
def research(request: ResearchRequest) -> ResearchResponse:
    """Research endpoint using the agent with RAG and tools.

    Performs comprehensive research using the knowledge base,
    web search, and summarization capabilities.

    Args:
        request: Research request with the query.

    Returns:
        Research response with answer and sources.

    Raises:
        HTTPException: On agent errors (500/503) or validation errors (422).
    """
    logger.info("Research request: %s...", request.query[:50])

    try:
        agent = get_research_agent()
        answer, sources = agent.research(request.query)
        return ResearchResponse(answer=answer, sources=sources)
    except TimeoutError as e:
        logger.error("Research timeout: %s", e)
        raise HTTPException(
            status_code=503,
            detail={"detail": "Research timed out", "error_code": "TIMEOUT"},
        )
    except Exception as e:
        logger.error("Research failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"detail": str(e), "error_code": "AGENT_ERROR"},
        )


@router.post(
    "/documents",
    response_model=DocumentResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
def add_document(request: DocumentRequest) -> DocumentResponse:
    """Add a document to the knowledge base.

    Documents are vectorized and stored for retrieval by the research agent.

    Args:
        request: Document request with title and content.

    Returns:
        Document response confirming the operation.

    Raises:
        HTTPException: On storage errors (500) or validation errors (422).
    """
    logger.info("Adding document: %s", request.title)

    try:
        agent = get_research_agent()
        count = agent.add_to_knowledge_base([{"title": request.title, "content": request.content}])

        if count > 0:
            return DocumentResponse(
                status="success",
                message=f"Document '{request.title}' added to knowledge base",
            )
        else:
            return DocumentResponse(
                status="warning",
                message="Document was not added (possibly empty content)",
            )
    except Exception as e:
        logger.error("Add document failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"detail": str(e), "error_code": "STORAGE_ERROR"},
        )
