from fastapi import APIRouter

from app.schemas.health import StatusResponse
from app.services.telemetry_service import telemetry_service

router = APIRouter(tags=["status"])


@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    return telemetry_service.get_status()