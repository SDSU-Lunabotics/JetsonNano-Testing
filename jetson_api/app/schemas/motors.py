from __future__ import annotations

from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from .common import Fault, MotorId


class MotorHealth(BaseModel):
    motor_id: MotorId
    enabled: Optional[bool] = None
    faults: Optional[List[Fault]] = None
    current_a: Optional[float] = None
    torque_nm: Optional[float] = None
    rpm: Optional[float] = None
    last_update_ms: Optional[int] = None


class MotorsStatusResponse(BaseModel):
    timestamp_ms: int
    motors: List[MotorHealth]


MotorCommandMode = Literal["percent", "rpm", "stop"]


class MotorCommandRequest(BaseModel):
    mode: MotorCommandMode
    value: Optional[float] = None
    duration_ms: Optional[int] = Field(default=None, ge=50)


class MotorCommandResponse(BaseModel):
    ok: bool
    motor_id: str
    timestamp_ms: int