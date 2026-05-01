import asyncio
import time
from typing import Iterator, Optional

from fastapi import APIRouter, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.core.settings import settings
from app.schemas.camera import (
    CameraStatus,
    CameraHeartbeatRequest,
    CameraHeartbeatResponse,
    CameraFrameIngestResponse,
    CameraModeRequest,
    CameraModeResponse,
    CameraWsMessage,
)
from app.services.camera_service import camera_service

router = APIRouter(prefix="/camera", tags=["camera"])


_ONE_PX_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)
_ONE_PX_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
    b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xcf \xff\xd9"
)


def _status_message(event_type: str) -> dict:
    msg = CameraWsMessage(
        type=event_type,  # type: ignore[arg-type]
        timestamp_ms=int(time.time() * 1000),
        camera=camera_service.get_cached_status(),
    )
    return msg.model_dump()


@router.get("/status", response_model=CameraStatus)
def get_camera_status() -> CameraStatus:
    return camera_service.get_cached_status()


@router.get("/snapshot")
def get_snapshot() -> Response:
    png = camera_service.get_snapshot_bytes() or _ONE_PX_PNG
    return Response(content=png, media_type="image/png")


@router.get("/snapshot.jpg")
def get_snapshot_jpeg() -> Response:
    jpeg = camera_service.get_latest_jpeg_bytes() or _ONE_PX_JPEG
    return Response(content=jpeg, media_type="image/jpeg")


def _mjpeg_stream(frame_interval_ms: int) -> Iterator[bytes]:
    sleep_s = max(frame_interval_ms, 1) / 1000.0
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    last_frame = _ONE_PX_JPEG

    while True:
        frame = camera_service.get_latest_jpeg_bytes() or last_frame
        if frame:
            last_frame = frame
        yield boundary + last_frame + b"\r\n"
        time.sleep(sleep_s)


@router.get("/stream")
def get_stream(
    frame_interval_ms: int = Query(default=settings.camera_stream_interval_ms, ge=50, le=2000),
) -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_stream(frame_interval_ms),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.websocket("/ws")
async def ws_camera(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json(_status_message("status"))

    last_mode_version = camera_service.get_mode_version()
    last_frame_seq = 0
    timeout_ms = max(settings.camera_stream_interval_ms, 50)

    try:
        while True:
            mode_version = camera_service.get_mode_version()
            if mode_version != last_mode_version:
                await websocket.send_json(_status_message("mode_changed"))
                last_mode_version = mode_version

            last_frame_seq, frame = await asyncio.to_thread(
                camera_service.wait_for_frame,
                last_frame_seq,
                timeout_ms,
                "jpeg",
            )
            if frame:
                await websocket.send_bytes(frame)
                continue

            await websocket.send_json(_status_message("error"))
            await asyncio.sleep(timeout_ms / 1000.0)
    except WebSocketDisconnect:
        return


@router.post("/mode", response_model=CameraModeResponse)
def set_camera_mode(req: CameraModeRequest) -> CameraModeResponse:
    return camera_service.set_mode(req)


@router.post("/heartbeat", response_model=CameraHeartbeatResponse)
def post_camera_heartbeat(req: CameraHeartbeatRequest) -> CameraHeartbeatResponse:
    camera_service.update_external_status(req)
    return CameraHeartbeatResponse(
        ok=True,
        timestamp_ms=int(time.time() * 1000),
    )


@router.post("/frame", response_model=CameraFrameIngestResponse)
async def post_camera_frame(
    request: Request,
    source: str = Query(default="zed_ground_wall"),
    timestamp_ms: Optional[int] = Query(default=None, ge=0),
) -> CameraFrameIngestResponse:
    payload = await request.body()
    frame_seq = camera_service.ingest_external_frame(
        payload,
        backend="zed",
        source=source,
        source_timestamp_ms=timestamp_ms,
    )
    return CameraFrameIngestResponse(
        ok=True,
        timestamp_ms=int(time.time() * 1000),
        frame_seq=frame_seq,
    )
