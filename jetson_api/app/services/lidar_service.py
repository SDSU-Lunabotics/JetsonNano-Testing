from __future__ import annotations

import socket
import threading
import time
from typing import List, Optional

from app.core.settings import settings
from app.schemas.common import Fault, Heartbeat
from app.schemas.lidar import (
    LidarPreviewMessage,
    LidarMode,
    LidarStatusResponse,
    LidarMapInfoResponse,
    MapOrigin,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


class LidarService:
    """
    Jetson-side lidar state manager for the local Unitree bridge pipeline.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_seen_ms: Optional[int] = None
        self._map_info = LidarMapInfoResponse(
            timestamp_ms=_now_ms(),
            frame_id="map",
            width=500,
            height=500,
            resolution_m_per_px=0.05,
            origin=MapOrigin(x_m=0.0, y_m=0.0),
        )
        self._mode: Optional[LidarMode] = None
        self._points_per_sec: Optional[float] = None
        self._frame_id: Optional[str] = None
        self._faults: List[Fault] = []
        self._map_png: bytes = b""
        self._backend_state = "idle"
        self._last_error: Optional[str] = None

    def start(self) -> None:
        backend = settings.lidar_backend.lower()
        if backend in {"none", "disabled"}:
            with self._lock:
                self._backend_state = "disabled"
                self._last_error = "LiDAR backend disabled by configuration"
            return

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_backend,
            name="lidar-service",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def _set_runtime_error(self, message: Optional[str]) -> None:
        with self._lock:
            self._last_error = message
            self._backend_state = "error" if message else "running"

    def set_faults(self, faults: Optional[List[Fault]]) -> None:
        with self._lock:
            self._faults = list(faults or [])

    def _offline_fault(self, now: int) -> Optional[Fault]:
        if self._last_seen_ms is None:
            detail = self._last_error or "No LiDAR bridge activity detected yet"
            return Fault(
                code="LIDAR_OFFLINE",
                severity="warn",
                message=detail,
                source="lidar",
                timestamp_ms=now,
            )

        age_ms = now - self._last_seen_ms
        if age_ms >= settings.lidar_status_ttl_ms:
            detail = self._last_error or f"LiDAR data is stale ({age_ms} ms old)"
            return Fault(
                code="LIDAR_OFFLINE",
                severity="warn",
                message=detail,
                source="lidar",
                timestamp_ms=now,
            )
        return None

    def _status_faults(self, now: int) -> Optional[List[Fault]]:
        faults = list(self._faults)
        offline_fault = self._offline_fault(now)
        if offline_fault is not None:
            faults.append(offline_fault)
        return faults or None

    def get_status(self) -> LidarStatusResponse:
        now = _now_ms()
        with self._lock:
            last_seen_ms = self._last_seen_ms
            mode = self._mode
            points_per_sec = self._points_per_sec
            frame_id = self._frame_id
            faults = self._status_faults(now)
        connected = last_seen_ms is not None and (now - last_seen_ms) < settings.lidar_status_ttl_ms

        hb = Heartbeat(
            connected=connected,
            last_seen_ms=last_seen_ms,
            age_ms=(None if last_seen_ms is None else now - last_seen_ms),
        )

        return LidarStatusResponse(
            timestamp_ms=now,
            heartbeat=hb,
            mode=mode,
            points_per_sec=points_per_sec,
            frame_id=frame_id,
            faults=faults,
        )

    def get_map_info(self) -> LidarMapInfoResponse:
        with self._lock:
            map_info = self._map_info
            return LidarMapInfoResponse(
                timestamp_ms=_now_ms(),
                frame_id=map_info.frame_id,
                width=map_info.width,
                height=map_info.height,
                resolution_m_per_px=map_info.resolution_m_per_px,
                origin=map_info.origin,
            )

    def get_map_png(self) -> bytes:
        with self._lock:
            return self._map_png

    def get_preview_message(self, seq: int) -> LidarPreviewMessage:
        return LidarPreviewMessage(
            type="lidar_preview",
            seq=seq,
            timestamp_ms=_now_ms(),
        )

    def _mark_online(self) -> None:
        with self._lock:
            self._last_seen_ms = _now_ms()
            self._mode = settings.lidar_mode if settings.lidar_mode in {"2d", "3d"} else "3d"
            self._frame_id = settings.lidar_frame_id
            self._points_per_sec = None
            self._last_error = None
            self._backend_state = "running"

    def _run_backend(self) -> None:
        backend = settings.lidar_backend.lower()
        if backend in {"auto", "unitree"}:
            self._run_unitree_backend()
            return
        self._set_runtime_error(f"Unsupported LiDAR backend '{settings.lidar_backend}'")

    def _run_unitree_backend(self) -> None:
        self._set_runtime_error("Waiting for Unitree LiDAR bridge and visualization service")
        interval_s = max(settings.lidar_monitor_interval_ms, 100) / 1000.0

        while not self._stop_event.is_set():
            data_ready = self._port_open(settings.lidar_tcp_host, settings.lidar_data_port)
            command_ready = self._port_open(settings.lidar_tcp_host, settings.lidar_command_port)

            if data_ready and command_ready:
                self._mark_online()
            else:
                self._set_runtime_error(
                    self._unitree_status_message(
                        data_ready=data_ready,
                        command_ready=command_ready,
                    )
                )

            self._stop_event.wait(interval_s)

    def _port_open(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.35):
                return True
        except OSError:
            return False

    def _unitree_status_message(self, *, data_ready: bool, command_ready: bool) -> str:
        if not data_ready and not command_ready:
            return (
                f"Unitree LiDAR pipeline offline. Expected visualization on "
                f"{settings.lidar_tcp_host}:{settings.lidar_data_port} and bridge command server on "
                f"{settings.lidar_tcp_host}:{settings.lidar_command_port}"
            )
        if not data_ready:
            return (
                f"LiDAR visualization service is not listening on "
                f"{settings.lidar_tcp_host}:{settings.lidar_data_port}"
            )
        return (
            f"LiDAR bridge command server is not listening on "
            f"{settings.lidar_tcp_host}:{settings.lidar_command_port}"
        )


lidar_service = LidarService()
