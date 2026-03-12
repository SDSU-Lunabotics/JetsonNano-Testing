from __future__ import annotations

import time
from typing import Optional, List

from app.schemas.common import Fault, Heartbeat
from app.schemas.lidar import (
    LidarMode,
    LidarStatusResponse,
    LidarMapInfoResponse,
    MapOrigin,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


class LidarService:
    """
    Jetson-side lidar state manager.
    v1 returns placeholder map bytes and metadata.
    """

    def __init__(self) -> None:
        self._last_seen_ms: Optional[int] = None

        self._map_info = LidarMapInfoResponse(
            timestamp_ms=_now_ms(),
            frame_id="map",
            width=500,
            height=500,
            resolution_m_per_px=0.05,
            origin=MapOrigin(x_m=0.0, y_m=0.0),
        )

        self._mode: Optional[LidarMode] = "2d"
        self._points_per_sec: Optional[float] = None
        self._frame_id: Optional[str] = "map"
        self._faults: Optional[List[Fault]] = None

    def update_heartbeat(self) -> None:
        self._last_seen_ms = _now_ms()

    def set_mode(self, mode: Optional[LidarMode]) -> None:
        self._mode = mode

    def set_points_per_sec(self, pps: Optional[float]) -> None:
        self._points_per_sec = pps

    def set_faults(self, faults: Optional[List[Fault]]) -> None:
        self._faults = faults

    def get_status(self) -> LidarStatusResponse:
        now = _now_ms()
        connected = self._last_seen_ms is not None and (now - self._last_seen_ms) < 2000

        hb = Heartbeat(
            connected=connected,
            last_seen_ms=self._last_seen_ms,
            age_ms=(None if self._last_seen_ms is None else now - self._last_seen_ms),
        )

        return LidarStatusResponse(
            timestamp_ms=now,
            heartbeat=hb,
            mode=self._mode,
            points_per_sec=self._points_per_sec,
            frame_id=self._frame_id,
            faults=self._faults,
        )

    def get_map_info(self) -> LidarMapInfoResponse:
        return LidarMapInfoResponse(
            timestamp_ms=_now_ms(),
            frame_id=self._map_info.frame_id,
            width=self._map_info.width,
            height=self._map_info.height,
            resolution_m_per_px=self._map_info.resolution_m_per_px,
            origin=self._map_info.origin,
        )

    def get_map_png(self) -> bytes:
        return b""


lidar_service = LidarService()