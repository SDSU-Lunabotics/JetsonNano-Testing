from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np

from app.core.settings import settings
from app.schemas.camera import (
    CameraBackend,
    CameraHeartbeatRequest,
    CameraMode,
    CameraModeRequest,
    CameraModeResponse,
    CameraStatus,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


class CameraService:
    """
    Jetson-side camera service with a persistent capture worker.

    The worker keeps the selected camera backend open, continuously reads frames,
    and updates an in-memory latest-frame buffer. HTTP snapshots, HTTP multipart
    streaming, and websocket streaming all read from that shared buffer instead
    of reopening the device per request.
    """

    def __init__(self) -> None:
        self._mode: CameraMode = "manual"
        self._snapshot_interval_ms: Optional[int] = None
        self._connected = False
        self._backend: Optional[CameraBackend] = None
        self._streaming = False
        self._last_frame_ms: Optional[int] = None
        self._last_error: Optional[str] = None
        self._last_status_check_ms = 0
        self._recent_activity_grace_ms = max(settings.camera_status_ttl_ms * 5, 10000)

        self._state_lock = threading.Lock()
        self._frame_condition = threading.Condition(self._state_lock)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._last_frame_seq = 0
        self._mode_version = 0
        self._latest_raw_frame = None
        self._latest_jpeg: bytes = b""
        self._latest_png: bytes = b""
        self._latest_png_seq = 0
        self._external_last_seen_ms: Optional[int] = None
        self._external_backend: Optional[CameraBackend] = None
        self._external_streaming = False
        self._external_source: Optional[str] = None
        self._external_timestamp_ms: Optional[int] = None

    def _maybe_start_worker(self) -> None:
        if settings.camera_autostart:
            self.start()

    def start(self) -> None:
        with self._state_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._capture_loop,
                name="jetson-camera-worker",
                daemon=True,
            )
            self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._frame_condition:
            self._frame_condition.notify_all()

        worker = self._worker_thread
        if worker is not None:
            worker.join(timeout=2.0)

    def get_mode(self) -> Tuple[CameraMode, Optional[int]]:
        with self._state_lock:
            return self._mode, self._snapshot_interval_ms

    def get_mode_version(self) -> int:
        with self._state_lock:
            return self._mode_version

    def set_mode(self, req: CameraModeRequest) -> CameraModeResponse:
        ok = True
        if req.mode == "snapshot" and req.snapshot_interval_ms is None:
            req = CameraModeRequest(mode=req.mode, snapshot_interval_ms=1000)
            ok = False

        with self._frame_condition:
            self._mode = req.mode
            self._snapshot_interval_ms = req.snapshot_interval_ms if req.mode == "snapshot" else None
            self._mode_version += 1
            self._frame_condition.notify_all()

        self._maybe_start_worker()

        applied = CameraModeRequest(
            mode=req.mode,
            snapshot_interval_ms=self._snapshot_interval_ms,
        )
        return CameraModeResponse(
            ok=ok,
            applied=applied,
            timestamp_ms=_now_ms(),
        )

    def _mark_disconnected(self, error: Optional[str] = None) -> None:
        with self._frame_condition:
            self._connected = False
            self._streaming = False
            self._backend = None
            self._last_error = error
            self._last_frame_ms = None
            self._frame_condition.notify_all()

    def _mark_connected(self, backend: CameraBackend) -> None:
        with self._frame_condition:
            self._connected = True
            self._streaming = True
            self._backend = backend
            self._last_error = None
            self._frame_condition.notify_all()

    def _mark_in_use(self, backend: CameraBackend, error: Optional[str]) -> None:
        with self._frame_condition:
            self._connected = True
            self._streaming = True
            self._backend = backend
            if self._last_frame_ms is None:
                self._last_frame_ms = _now_ms()
            self._last_error = error
            self._frame_condition.notify_all()

    def _store_frame(self, frame, backend: CameraBackend, cv2_module) -> None:
        ok, encoded = cv2_module.imencode(
            ".jpg",
            frame,
            [int(cv2_module.IMWRITE_JPEG_QUALITY), int(settings.camera_jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("Failed to encode camera frame as JPEG")

        now = _now_ms()
        with self._frame_condition:
            self._latest_raw_frame = frame.copy()
            self._latest_jpeg = encoded.tobytes()
            self._latest_png = b""
            self._latest_png_seq = 0
            self._last_frame_ms = now
            self._last_frame_seq += 1
            self._connected = True
            self._streaming = True
            self._backend = backend
            self._last_error = None
            self._frame_condition.notify_all()

    def _error_implies_camera_in_use(self, error: Optional[str]) -> bool:
        if not error:
            return False

        normalized = error.lower().replace("_", " ")
        markers = (
            "busy",
            "in use",
            "already in use",
            "device in use",
            "device or resource busy",
            "resource busy",
            "already opened",
            "already used",
            "exclusive access",
        )
        return any(marker in normalized for marker in markers)

    def _has_recent_external_status(self, now_ms: Optional[int] = None) -> bool:
        with self._state_lock:
            if self._external_last_seen_ms is None:
                return False
            current_ms = _now_ms() if now_ms is None else now_ms
            return (current_ms - self._external_last_seen_ms) < self._recent_activity_grace_ms

    def update_external_status(self, req: CameraHeartbeatRequest) -> None:
        seen_ms = _now_ms()
        with self._frame_condition:
            self._external_last_seen_ms = seen_ms
            self._external_backend = req.backend
            self._external_streaming = bool(req.streaming)
            self._external_source = req.source
            self._external_timestamp_ms = req.timestamp_ms

            # If another process owns the camera, reflect that in status immediately.
            if self._last_frame_ms is None or (seen_ms - self._last_frame_ms) >= settings.camera_status_ttl_ms:
                self._connected = True
                self._backend = req.backend
                self._streaming = bool(req.streaming)
                self._last_frame_ms = req.timestamp_ms or seen_ms
                source = f" by {req.source}" if req.source else ""
                self._last_error = f"Camera is active{source}; API does not own the device"
            self._frame_condition.notify_all()

    def ingest_external_frame(
        self,
        frame_bytes: bytes,
        *,
        backend: CameraBackend,
        source: Optional[str] = None,
        source_timestamp_ms: Optional[int] = None,
    ) -> int:
        if not frame_bytes:
            raise ValueError("Camera frame payload is empty")

        raw_frame = None
        try:
            import cv2  # type: ignore

            encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
            raw_frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except Exception:
            raw_frame = None

        now = _now_ms()
        with self._frame_condition:
            self._latest_jpeg = frame_bytes
            self._latest_raw_frame = None if raw_frame is None else raw_frame.copy()
            self._latest_png = b""
            self._latest_png_seq = 0
            self._last_frame_ms = source_timestamp_ms or now
            self._last_frame_seq += 1
            self._connected = True
            self._streaming = True
            self._backend = backend
            self._last_error = None
            self._external_last_seen_ms = now
            self._external_backend = backend
            self._external_streaming = True
            self._external_source = source
            self._external_timestamp_ms = source_timestamp_ms
            self._frame_condition.notify_all()
            return self._last_frame_seq

    def _has_recent_activity(self) -> bool:
        with self._state_lock:
            return (
                self._last_frame_ms is not None
                and (_now_ms() - self._last_frame_ms) < max(settings.camera_status_ttl_ms, 1000)
            )

    def _probe_zed(self) -> bool:
        try:
            import pyzed.sl as sl  # type: ignore
        except Exception as exc:
            self._last_error = f"ZED SDK unavailable: {exc}"
            return False

        cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.camera_fps = settings.camera_capture_fps

        err = cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self._last_error = f"ZED open failed: {err}"
            return False

        try:
            runtime = sl.RuntimeParameters()
            image = sl.Mat()
            for _ in range(5):
                grab = cam.grab(runtime)
                if grab == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(image, sl.VIEW.LEFT)
                    frame = image.get_data()
                    if frame is not None:
                        self._mark_connected("zed")
                        return True
            self._last_error = "ZED connected but no frames were available"
            return False
        finally:
            cam.close()

    def _probe_opencv(self) -> bool:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            self._last_error = f"OpenCV unavailable: {exc}"
            return False

        cap = cv2.VideoCapture(settings.camera_device_index)
        if not cap.isOpened():
            self._last_error = f"OpenCV could not open camera index {settings.camera_device_index}"
            cap.release()
            return False

        try:
            ok, _frame = cap.read()
            if ok:
                self._mark_connected("opencv")
                return True
            self._last_error = f"OpenCV camera index {settings.camera_device_index} returned no frame"
            return False
        finally:
            cap.release()

    def refresh_status(self, force: bool = False) -> None:
        self._maybe_start_worker()
        now = _now_ms()
        if not force and (now - self._last_status_check_ms) < settings.camera_status_ttl_ms:
            return

        self._last_status_check_ms = now
        if self._has_recent_activity():
            with self._state_lock:
                self._connected = True
                self._streaming = True
                self._last_error = None
            return

        if self._has_recent_external_status(now):
            with self._state_lock:
                self._connected = True
                self._backend = self._external_backend
                self._streaming = bool(self._external_streaming)
                if self._external_timestamp_ms is not None:
                    self._last_frame_ms = self._external_timestamp_ms
                elif self._external_last_seen_ms is not None:
                    self._last_frame_ms = self._external_last_seen_ms
                source = f" by {self._external_source}" if self._external_source else ""
                self._last_error = f"Camera is active{source}; API does not own the device"
            return

        backend = settings.camera_backend.lower()
        if backend in ("auto", "zed") and self._probe_zed():
            return
        if backend in ("auto", "zed") and self._error_implies_camera_in_use(self._last_error):
            self._mark_in_use("zed", self._last_error)
            return
        if backend in ("auto", "opencv") and self._probe_opencv():
            return
        if backend in ("auto", "opencv") and self._error_implies_camera_in_use(self._last_error):
            self._mark_in_use("opencv", self._last_error)
            return
        if backend not in ("auto", "zed", "opencv"):
            self._mark_disconnected(f"Unsupported camera backend '{settings.camera_backend}'")
            return
        if not self._has_recent_activity():
            self._mark_disconnected(self._last_error)

    def get_status(self) -> CameraStatus:
        self.refresh_status()
        return self.get_cached_status()

    def get_cached_status(self) -> CameraStatus:
        with self._state_lock:
            return CameraStatus(
                connected=self._connected,
                backend=self._backend,
                streaming=self._streaming,
                mode=self._mode,
                snapshot_interval_ms=self._snapshot_interval_ms,
                last_frame_ms=self._last_frame_ms,
                source=self._external_source,
                source_timestamp_ms=self._external_timestamp_ms,
                error=self._last_error,
            )

    def _get_latest_raw_frame_copy(self):
        with self._state_lock:
            if self._latest_raw_frame is None:
                return None, 0
            return self._latest_raw_frame.copy(), self._last_frame_seq

    def get_snapshot_bytes(self) -> bytes:
        self._maybe_start_worker()
        raw_frame, frame_seq = self._get_latest_raw_frame_copy()
        if raw_frame is None:
            return b""

        with self._state_lock:
            if self._latest_png and self._latest_png_seq == frame_seq:
                return self._latest_png

        try:
            import cv2  # type: ignore
        except Exception as exc:
            self._mark_disconnected(f"OpenCV is required for PNG snapshots: {exc}")
            return b""

        ok, encoded = cv2.imencode(".png", raw_frame)
        if not ok:
            self._mark_disconnected("Failed to encode camera frame as PNG")
            return b""

        png = encoded.tobytes()
        with self._state_lock:
            if self._latest_png_seq != frame_seq:
                self._latest_png = png
                self._latest_png_seq = frame_seq
            return self._latest_png

    def get_latest_jpeg_bytes(self) -> bytes:
        self._maybe_start_worker()
        with self._state_lock:
            return self._latest_jpeg

    def wait_for_frame(
        self,
        last_seq: int,
        timeout_ms: Optional[int] = None,
        image_format: str = "jpeg",
    ) -> Tuple[int, bytes]:
        self._maybe_start_worker()
        timeout_s = None if timeout_ms is None else max(timeout_ms, 0) / 1000.0
        deadline = None if timeout_s is None else time.monotonic() + timeout_s

        with self._frame_condition:
            while not self._stop_event.is_set():
                if self._last_frame_seq > last_seq:
                    payload = self._latest_jpeg if image_format == "jpeg" else self._latest_png
                    if payload:
                        return self._last_frame_seq, payload
                    if image_format == "png":
                        break

                if timeout_s is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._frame_condition.wait(timeout=remaining)
                else:
                    self._frame_condition.wait()

        if image_format == "png":
            png = self.get_snapshot_bytes()
            with self._state_lock:
                return self._last_frame_seq, png
        with self._state_lock:
            return self._last_frame_seq, self._latest_jpeg

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._has_recent_external_status():
                with self._state_lock:
                    self._connected = True
                    self._backend = self._external_backend
                    self._streaming = False
                    if self._external_timestamp_ms is not None:
                        self._last_frame_ms = self._external_timestamp_ms
                    elif self._external_last_seen_ms is not None:
                        self._last_frame_ms = self._external_last_seen_ms
                    source = f" by {self._external_source}" if self._external_source else ""
                    self._last_error = f"Camera is active{source}; API does not own the device"
                time.sleep(max(settings.camera_worker_retry_ms, 100) / 1000.0)
                continue

            backend = settings.camera_backend.lower()
            candidates = []
            if backend == "auto":
                candidates = ["zed", "opencv"]
            elif backend in ("zed", "opencv"):
                candidates = [backend]
            else:
                self._mark_disconnected(f"Unsupported camera backend '{settings.camera_backend}'")
                time.sleep(max(settings.camera_worker_retry_ms, 100) / 1000.0)
                continue

            last_error: Optional[str] = None
            session_started = False

            for candidate in candidates:
                if self._stop_event.is_set():
                    break

                try:
                    if candidate == "zed":
                        self._run_zed_capture_loop()
                    else:
                        self._run_opencv_capture_loop()
                    session_started = True
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if self._error_implies_camera_in_use(last_error):
                        self._mark_in_use(candidate, last_error)  # type: ignore[arg-type]
                        session_started = True
                        break

            if self._stop_event.is_set():
                break

            if not session_started:
                self._mark_disconnected(last_error or "No supported camera backend is available")

            time.sleep(max(settings.camera_worker_retry_ms, 100) / 1000.0)

    def _run_zed_capture_loop(self) -> None:
        try:
            import cv2  # type: ignore
            import pyzed.sl as sl  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"ZED capture dependencies unavailable: {exc}") from exc

        cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.camera_fps = settings.camera_capture_fps

        err = cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        try:
            runtime = sl.RuntimeParameters()
            image = sl.Mat()

            while not self._stop_event.is_set():
                grab = cam.grab(runtime)
                if grab != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(f"Failed to grab ZED frame: {grab}")

                cam.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()
                if frame is None:
                    continue

                self._store_frame(frame, "zed", cv2)
        finally:
            cam.close()

    def _run_opencv_capture_loop(self) -> None:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"OpenCV is required for camera capture: {exc}") from exc

        cap = cv2.VideoCapture(settings.camera_device_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open camera index {settings.camera_device_index}")

        try:
            cap.set(cv2.CAP_PROP_FPS, float(settings.camera_capture_fps))
        except Exception:
            pass

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"Failed to read frame from camera index {settings.camera_device_index}"
                    )

                self._store_frame(frame, "opencv", cv2)
        finally:
            cap.release()


camera_service = CameraService()
