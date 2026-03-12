from __future__ import annotations

from typing import Literal, Optional, List
from pydantic import BaseModel, Field

from .common import Heartbeat, Fault


LidarMode = Literal["2d", "3d"]


class LidarStatusResponse(BaseModel):
    timestamp_ms: int
    heartbeat: Heartbeat
    mode: Optional[LidarMode] = None
    points_per_sec: Optional[float] = None
    frame_id: Optional[str] = None
    faults: Optional[List[Fault]] = None


class MapOrigin(BaseModel):
    x_m: float
    y_m: float


class LidarMapInfoResponse(BaseModel):
    timestamp_ms: int
    frame_id: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    resolution_m_per_px: float = Field(gt=0)
    origin: MapOrigin


class LidarPreviewMessage(BaseModel):
    type: Literal["lidar_preview"]
    seq: int
    timestamp_ms: int