from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

from app.schemas.map import MapStreamStatus


def _now_ms() -> int:
    return int(time.time() * 1000)


class MapService:
    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._frame_condition = threading.Condition(self._state_lock)
        self._latest_jpeg: bytes = b""
        self._frame_seq = 0
        self._last_frame_ms: Optional[int] = None
        self._source_timestamp_ms: Optional[int] = None
        self._source: Optional[str] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None

    def ingest_jpeg(
        self,
        frame_bytes: bytes,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        source: Optional[str] = None,
        source_timestamp_ms: Optional[int] = None,
    ) -> int:
        if not frame_bytes:
            raise ValueError("Map frame payload is empty")

        with self._frame_condition:
            self._latest_jpeg = frame_bytes
            self._frame_seq += 1
            self._last_frame_ms = _now_ms()
            self._source_timestamp_ms = source_timestamp_ms
            self._source = source
            self._width = width
            self._height = height
            self._frame_condition.notify_all()
            return self._frame_seq

    def get_latest_jpeg_bytes(self) -> bytes:
        with self._state_lock:
            return self._latest_jpeg

    def get_status(self) -> MapStreamStatus:
        with self._state_lock:
            return MapStreamStatus(
                available=bool(self._latest_jpeg),
                source=self._source,
                width=self._width,
                height=self._height,
                last_frame_ms=self._last_frame_ms,
                source_timestamp_ms=self._source_timestamp_ms,
                frame_seq=self._frame_seq,
            )

    def wait_for_frame(self, last_frame_seq: int, timeout_ms: int) -> Tuple[int, bytes]:
        timeout_s = max(timeout_ms, 1) / 1000.0
        deadline = time.monotonic() + timeout_s

        with self._frame_condition:
            while self._frame_seq <= last_frame_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._frame_seq, b""
                self._frame_condition.wait(timeout=remaining)

            return self._frame_seq, self._latest_jpeg


map_service = MapService()
