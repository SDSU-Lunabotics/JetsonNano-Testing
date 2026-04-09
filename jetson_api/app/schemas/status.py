from __future__ import annotations

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel

from .common import BatteryStatus, ControllerStatus, Heartbeat, LinkStats, ControlMode


class RoverStatus(BaseModel):
    heartbeat: Heartbeat
    jetson: Optional[Heartbeat] = None
    roborio: Optional[Heartbeat] = None
    lidar: Optional[Heartbeat] = None


class ControlStatus(BaseModel):
    armed: bool
    estop: bool
    mode: ControlMode
    autonomy_running: bool
    controller: ControllerStatus


class TelemetryMessage(BaseModel):
    type: Literal["telemetry"] = "telemetry"
    seq: int
    timestamp_ms: int
    data: Dict[str, Any]