"""Health check endpoint handlers."""

from fastapi import APIRouter
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring and load balancers.

    Returns:
        HealthResponse: Simple status indicating the service is healthy
    """
    return HealthResponse(status="healthy")
