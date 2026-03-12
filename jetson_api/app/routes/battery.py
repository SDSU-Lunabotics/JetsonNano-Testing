from fastapi import APIRouter

from app.schemas.common import BatteryStatus
from app.services.battery_service import battery_service

router = APIRouter(prefix="/battery", tags=["battery"])


@router.get("", response_model=BatteryStatus)
def get_battery() -> BatteryStatus:
    return battery_service.get_battery_status()