from __future__ import annotations

from typing import Literal, Optional, List
from pydantic import BaseModel, Field

from .common import Fault, LinkStats


NetworkTarget = Literal["jetson", "roborio", "rover", "camera", "lidar"]


class NetworkVerifyRequest(BaseModel):
    target: Optional[NetworkTarget] = "rover"
    timeout_ms: Optional[int] = Field(default=5000, ge=100)


class NetworkVerifyResponse(BaseModel):
    ok: bool
    timestamp_ms: int
    link: LinkStats
    message: Optional[str] = None


class NetworkStatusResponse(BaseModel):
    timestamp_ms: int
    link: LinkStats
    ok: bool
    faults: Optional[List[Fault]] = None