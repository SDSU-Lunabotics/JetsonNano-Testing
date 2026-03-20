from __future__ import annotations

import time
from typing import Optional, Tuple

from app.core.settings import settings
from app.schemas.camera import (
    CameraBackend,
    CameraMode,
    CameraModeRequest,
    CameraModeResponse,
    CameraStatus,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


class CameraService:
    """
    Jetson-side camera state manager.
    Supports direct camera attachment to the Jetson through either:
    - ZED SDK (`pyzed.sl`) when available
    - OpenCV (`cv2`) device capture when available
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

    def _mark_disconnected(self, error: Optional[str] = None) -> None:
        self._connected = False
        self._streaming = False
        self._backend = None
        self._last_error = error

    def _mark_connected(self, backend: CameraBackend) -> None:
        self._connected = True
        self._streaming = True
        self._backend = backend
        self._last_frame_ms = _now_ms()
        self._last_error = None

    def _has_recent_activity(self) -> bool:
        return (
            self._last_frame_ms is not None
            and (_now_ms() - self._last_frame_ms) < self._recent_activity_grace_ms
        )

    def _preserve_recent_connection(self, error: Optional[str]) -> bool:
        if not self._has_recent_activity() or self._backend is None:
            return False

        # A busy camera can fail a second open() even though the active pipeline is healthy.
        self._connected = True
        self._streaming = True
        self._last_error = error
        return True

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
        init_params.camera_fps = 30

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
        now = _now_ms()
        if not force and (now - self._last_status_check_ms) < settings.camera_status_ttl_ms:
            return

        self._last_status_check_ms = now
        backend = settings.camera_backend.lower()

        if backend in ("auto", "zed") and self._probe_zed():
            return
        if backend in ("auto", "opencv") and self._probe_opencv():
            return

        if backend not in ("auto", "zed", "opencv"):
            self._mark_disconnected(f"Unsupported camera backend '{settings.camera_backend}'")
            return

        if self._preserve_recent_connection(self._last_error):
            return

        self._mark_disconnected(self._last_error)

    def get_status(self) -> CameraStatus:
        self.refresh_status()
        return CameraStatus(
            connected=self._connected,
            backend=self._backend,
            streaming=self._streaming,
            mode=self._mode,
            snapshot_interval_ms=self._snapshot_interval_ms,
            last_frame_ms=self._last_frame_ms,
            error=self._last_error,
        )

    def _snapshot_zed(self) -> bytes:
        import pyzed.sl as sl  # type: ignore

        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"OpenCV is required for ZED snapshots: {exc}") from exc

        cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.camera_fps = 30

        err = cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        try:
            runtime = sl.RuntimeParameters()
            image = sl.Mat()
            for _ in range(10):
                grab = cam.grab(runtime)
                if grab == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(image, sl.VIEW.LEFT)
                    frame = image.get_data()
                    if frame is None:
                        continue
                    ok, encoded = cv2.imencode(".png", frame)
                    if not ok:
                        raise RuntimeError("Failed to encode ZED frame as PNG")
                    self._mark_connected("zed")
                    return encoded.tobytes()
            raise RuntimeError("Timed out waiting for a ZED frame")
        finally:
            cam.close()

    def _snapshot_opencv(self) -> bytes:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"OpenCV is required for camera snapshots: {exc}") from exc

        cap = cv2.VideoCapture(settings.camera_device_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open camera index {settings.camera_device_index}")

        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame from camera index {settings.camera_device_index}")
            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                raise RuntimeError("Failed to encode camera frame as PNG")
            self._mark_connected("opencv")
            return encoded.tobytes()
        finally:
            cap.release()

    def get_snapshot_bytes(self) -> bytes:
        backend = settings.camera_backend.lower()
        last_error: Optional[str] = None

        if backend in ("auto", "zed"):
            try:
                return self._snapshot_zed()
            except Exception as exc:
                last_error = str(exc)

        if backend in ("auto", "opencv"):
            try:
                return self._snapshot_opencv()
            except Exception as exc:
                last_error = str(exc)

        self._mark_disconnected(last_error or "No supported camera backend is available")
        return b""


camera_service = CameraService()
