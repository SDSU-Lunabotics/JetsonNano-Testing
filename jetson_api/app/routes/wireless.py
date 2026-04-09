from fastapi import APIRouter, HTTPException

from app.schemas.wireless import (
    WirelessConfigUpdateRequest,
    WirelessConfigUpdateResponse,
    WirelessStatusResponse,
)
from app.services.wireless_service import wireless_service

router = APIRouter(prefix="/wireless", tags=["wireless"])


@router.get("/status", response_model=WirelessStatusResponse)
def get_wireless_status() -> WirelessStatusResponse:
    return wireless_service.get_status()


@router.post("/config", response_model=WirelessConfigUpdateResponse)
def update_wireless_config(req: WirelessConfigUpdateRequest) -> WirelessConfigUpdateResponse:
    try:
        return wireless_service.update_team_ssid(req.team_ssid)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
