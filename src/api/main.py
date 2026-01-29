"""FastAPI application for the Research Assistant."""

import logging
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes.chat import router as chat_router
from src.api.routes.research import router as research_router
from src.models.schemas import HealthResponse

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Assistant API",
    description="AI-powered research assistant with RAG and agent capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration - allow localhost:3000 for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(research_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler to prevent stack trace leakage.

    Args:
        request: The incoming request.
        exc: The exception that was raised.

    Returns:
        JSON response with sanitized error details.
    """
    logger.error("Unhandled exception: %s", exc, exc_info=True)

    # Don't expose internal details in production
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again later.",
            "error_code": "INTERNAL_ERROR",
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
)
def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns the current health status of the service.
    Use this endpoint for load balancer health checks.

    Returns:
        Health response with status "healthy".
    """
    return HealthResponse(status="healthy")


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    """Root endpoint redirect to docs.

    Returns:
        Redirect message to API documentation.
    """
    return {"message": "Research Assistant API - Visit /docs for documentation"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
