from __future__ import annotations

import math
import socket
import struct
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

from app.core.settings import settings
from app.schemas.common import Fault, Heartbeat
from app.schemas.lidar import (
    LidarPoint,
    LidarPreviewMessage,
    LidarMode,
    LidarStatusResponse,
    LidarMapInfoResponse,
    MapOrigin,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


POINT_BYTES = 12


class LidarService:
    """
    Jetson-side lidar state manager for the local Unitree bridge pipeline.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._receiver_thread: Optional[threading.Thread] = None
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
        self._latest_points: List[Tuple[float, float, float]] = []
        self._latest_point_count: int = 0
        self._latest_truncated: bool = False
        self._recv_timestamps_s: Deque[float] = deque()
        self._last_launch_attempt_ms: Optional[int] = None
        self._last_launch_kind: Optional[str] = None

    def start(self) -> None:
        backend = settings.lidar_backend.lower()
        if backend in {"none", "disabled"}:
            with self._lock:
                self._backend_state = "disabled"
                self._last_error = "LiDAR backend disabled by configuration"
            return

        receiver_alive = self._receiver_thread and self._receiver_thread.is_alive()
        monitor_alive = self._monitor_thread and self._monitor_thread.is_alive()
        if receiver_alive or monitor_alive:
            return

        self._stop_event.clear()
        self._receiver_thread = threading.Thread(
            target=self._run_receiver,
            name="lidar-receiver",
            daemon=True,
        )
        self._receiver_thread.start()
        self._monitor_thread = threading.Thread(
            target=self._run_backend_monitor,
            name="lidar-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for thread in (self._receiver_thread, self._monitor_thread):
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        self._receiver_thread = None
        self._monitor_thread = None

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
            backend_state = self._backend_state
            last_error = self._last_error
            last_launch_kind = self._last_launch_kind
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
            backend_state=backend_state,
            last_error=last_error,
            last_launch_kind=last_launch_kind,
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
        with self._lock:
            points = [LidarPoint(x=x, y=y, z=z) for x, y, z in self._latest_points]
            frame_id = self._frame_id
            points_per_sec = self._points_per_sec
            point_count = self._latest_point_count
            truncated = self._latest_truncated
        return LidarPreviewMessage(
            type="lidar_preview",
            seq=seq,
            timestamp_ms=_now_ms(),
            frame_id=frame_id,
            point_count=point_count,
            points_per_sec=points_per_sec,
            truncated=truncated,
            points=points,
        )

    def _mark_online(self) -> None:
        with self._lock:
            self._last_seen_ms = _now_ms()
            self._mode = settings.lidar_mode if settings.lidar_mode in {"2d", "3d"} else "3d"
            self._frame_id = settings.lidar_frame_id
            self._last_error = None
            self._backend_state = "running"

    def _run_backend_monitor(self) -> None:
        backend = settings.lidar_backend.lower()
        if backend in {"auto", "unitree"}:
            self._run_unitree_backend()
            return
        self._set_runtime_error(f"Unsupported LiDAR backend '{settings.lidar_backend}'")

    def _run_receiver(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        timeout_s = max(settings.lidar_socket_timeout_ms, 100) / 1000.0
        server.settimeout(timeout_s)
        try:
            server.bind((settings.lidar_tcp_host, settings.lidar_data_port))
            server.listen(1)
        except OSError as exc:
            self._set_runtime_error(
                f"LiDAR receiver failed to bind on "
                f"{settings.lidar_tcp_host}:{settings.lidar_data_port}: {exc}"
            )
            server.close()
            return

        while not self._stop_event.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                conn.settimeout(timeout_s)
                self._receive_points(conn)

        server.close()

    def _receive_points(self, conn: socket.socket) -> None:
        buffer = b""
        batch: List[Tuple[float, float, float]] = []
        batch_started_ms = _now_ms()
        max_points = max(settings.lidar_preview_max_points, 1)

        while not self._stop_event.is_set():
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                if batch:
                    self._publish_batch(batch, batch_started_ms, max_points)
                    batch = []
                    batch_started_ms = _now_ms()
                continue
            except OSError:
                break

            if not chunk:
                break

            now_ms = _now_ms()
            buffer += chunk
            while len(buffer) >= POINT_BYTES:
                x, y, z = struct.unpack_from("<fff", buffer)
                buffer = buffer[POINT_BYTES:]
                if not all(math.isfinite(v) for v in (x, y, z)):
                    continue
                batch.append((float(x), float(y), float(z)))

            if batch:
                self._publish_batch(batch, now_ms, max_points)
                batch = []
                batch_started_ms = now_ms

    def _publish_batch(
        self,
        batch: List[Tuple[float, float, float]],
        timestamp_ms: int,
        max_points: int,
    ) -> None:
        timestamp_s = timestamp_ms / 1000.0
        with self._lock:
            self._last_seen_ms = timestamp_ms
            self._mode = settings.lidar_mode if settings.lidar_mode in {"2d", "3d"} else "3d"
            self._frame_id = settings.lidar_frame_id
            self._backend_state = "running"
            self._last_error = None
            self._latest_point_count = len(batch)
            self._latest_truncated = len(batch) > max_points
            self._latest_points = list(batch[:max_points])
            self._recv_timestamps_s.append(timestamp_s)
            cutoff = timestamp_s - 1.0
            while self._recv_timestamps_s and self._recv_timestamps_s[0] < cutoff:
                self._recv_timestamps_s.popleft()
            self._points_per_sec = float(len(self._recv_timestamps_s))

    def _run_unitree_backend(self) -> None:
        self._set_runtime_error("Waiting for Unitree LiDAR bridge")
        interval_s = max(settings.lidar_monitor_interval_ms, 100) / 1000.0

        while not self._stop_event.is_set():
            command_ready = self._port_open(settings.lidar_tcp_host, settings.lidar_command_port)
            now = _now_ms()
            with self._lock:
                has_recent_points = (
                    self._last_seen_ms is not None and
                    (now - self._last_seen_ms) < settings.lidar_status_ttl_ms
                )
                bind_failed = self._last_error is not None and "failed to bind" in self._last_error.lower()
                should_restart = self._last_seen_ms is not None and not has_recent_points
                can_launch = (
                    settings.lidar_autostart and
                    not bind_failed and
                    (
                        self._last_launch_attempt_ms is None or
                        (now - self._last_launch_attempt_ms) >= settings.lidar_autostart_cooldown_ms
                    )
                )

            if bind_failed:
                self._stop_event.wait(interval_s)
                continue

            if has_recent_points and command_ready:
                with self._lock:
                    self._last_launch_kind = None
                self._mark_online()
            else:
                if can_launch:
                    launch_kind = "restart_lidar" if should_restart else "start_lidar"
                    self._launch_lidar_script(launch_kind, now)
                self._set_runtime_error(
                    self._unitree_status_message(
                        has_recent_points=has_recent_points,
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

    def _unitree_status_message(self, *, has_recent_points: bool, command_ready: bool) -> str:
        if not has_recent_points and not command_ready:
            return (
                f"Unitree LiDAR pipeline offline. Expected point batches on "
                f"{settings.lidar_tcp_host}:{settings.lidar_data_port} and bridge command server on "
                f"{settings.lidar_tcp_host}:{settings.lidar_command_port}"
            )
        if not has_recent_points:
            return (
                f"LiDAR point stream is not active on "
                f"{settings.lidar_tcp_host}:{settings.lidar_data_port}"
            )
        return (
            f"LiDAR bridge command server is not listening on "
            f"{settings.lidar_tcp_host}:{settings.lidar_command_port}"
        )

    def _launch_lidar_script(self, name: str, attempted_ms: int) -> None:
        script_name = f"{name}.sh"
        script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
        if not script_path.exists():
            self._set_runtime_error(f"LiDAR script not found: {script_path}")
            with self._lock:
                self._last_launch_attempt_ms = attempted_ms
                self._last_launch_kind = f"{name}:missing"
            return

        try:
            result = subprocess.run(
                [str(script_path)],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except Exception as exc:
            self._set_runtime_error(f"Failed to run {name}: {exc}")
            with self._lock:
                self._last_launch_attempt_ms = attempted_ms
                self._last_launch_kind = f"{name}:failed"
            return

        combined_output = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part and part.strip()
        )
        if result.returncode != 0:
            detail = combined_output or f"{name} exited with code {result.returncode}"
            self._set_runtime_error(f"{name} failed: {detail}")
            with self._lock:
                self._last_launch_attempt_ms = attempted_ms
                self._last_launch_kind = f"{name}:failed"
            return

        with self._lock:
            self._last_launch_attempt_ms = attempted_ms
            self._last_launch_kind = name


lidar_service = LidarService()
