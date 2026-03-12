from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


CameraMode = Literal["manual", "snapshot"]


class CameraModeRequest(BaseModel):
    mode: CameraMode
    snapshot_interval_ms: Optional[int] = Field(default=None, ge=100)


class CameraModeResponse(BaseModel):
    ok: bool
    applied: CameraModeRequest
    timestamp_ms: int