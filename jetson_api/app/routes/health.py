from fastapi import APIRouter

from app.schemas.health import HealthResponse, TimeResponse
from app.services.state_service import now_ms

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(timestamp_ms=now_ms())


@router.get("/time", response_model=TimeResponse)
def time_status() -> TimeResponse:
    return TimeResponse(timestamp_ms=now_ms())


# Backwards-compatible liveness endpoint.
@router.get("/ping")
def ping():
    return {"ok": True, "service": "jetson", "timestamp_ms": now_ms()}
