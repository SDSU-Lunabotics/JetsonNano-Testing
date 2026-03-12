import os
from pydantic import BaseModel

from .common import LinkStats, BatteryStatus
from .status import RoverStatus, ControlStatus
from .wireless import WirelessStatusResponse

VERSION = os.getenv("APP_VERSION", "dev")


class HealthResponse(BaseModel):
    ok: bool = True
    service: str = "jetson-lunabotics-api"
    version: str = VERSION
    timestamp_ms: int


class TimeResponse(BaseModel):
    timestamp_ms: int


class StatusResponse(BaseModel):
    timestamp_ms: int
    rover: RoverStatus
    link: LinkStats
    battery: BatteryStatus
    control: ControlStatus
    wireless: WirelessStatusResponse