from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


CameraMode = Literal["manual", "snapshot"]
CameraBackend = Literal["auto", "zed", "opencv"]
CameraStreamEvent = Literal["status", "mode_changed", "error"]


class CameraModeRequest(BaseModel):
    mode: CameraMode
    snapshot_interval_ms: Optional[int] = Field(default=None, ge=100)


class CameraModeResponse(BaseModel):
    ok: bool
    applied: CameraModeRequest
    timestamp_ms: int


class CameraStatus(BaseModel):
    connected: bool
    backend: Optional[CameraBackend] = None
    streaming: bool = False
    mode: CameraMode
    snapshot_interval_ms: Optional[int] = None
    last_frame_ms: Optional[int] = None
    error: Optional[str] = None


class CameraWsMessage(BaseModel):
    type: CameraStreamEvent
    timestamp_ms: int
    camera: CameraStatus
