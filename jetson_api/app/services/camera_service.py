from __future__ import annotations

import time
from typing import Optional, Tuple

from app.schemas.camera import CameraMode, CameraModeRequest, CameraModeResponse, CameraStatus


def _now_ms() -> int:
    return int(time.time() * 1000)


class CameraService:
    """
    Jetson-side camera state manager.
    v1 still returns placeholder snapshot bytes unless you plug in a real camera pipeline.
    """

    def __init__(self) -> None:
        self._mode: CameraMode = "manual"
        self._snapshot_interval_ms: Optional[int] = None

    def get_mode(self) -> Tuple[CameraMode, Optional[int]]:
        return self._mode, self._snapshot_interval_ms

    def set_mode(self, req: CameraModeRequest) -> CameraModeResponse:
        if req.mode == "snapshot" and req.snapshot_interval_ms is None:
            applied = CameraModeRequest(mode=req.mode, snapshot_interval_ms=1000)
            self._mode = "snapshot"
            self._snapshot_interval_ms = 1000
            return CameraModeResponse(
                ok=False,
                applied=applied,
                timestamp_ms=_now_ms(),
            )

        self._mode = req.mode
        self._snapshot_interval_ms = req.snapshot_interval_ms if req.mode == "snapshot" else None

        applied = CameraModeRequest(
            mode=self._mode,
            snapshot_interval_ms=self._snapshot_interval_ms,
        )
        return CameraModeResponse(
            ok=True,
            applied=applied,
            timestamp_ms=_now_ms(),
        )

    def get_status(self) -> CameraStatus:
        return CameraStatus(
            mode=self._mode,
            snapshot_interval_ms=self._snapshot_interval_ms,
        )

    def get_snapshot_bytes(self) -> bytes:
        """
        Replace later with real camera capture bytes.
        """
        return b""


camera_service = CameraService()
