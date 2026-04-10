import time
from typing import Iterator, Optional

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import StreamingResponse

from app.core.settings import settings
from app.schemas.map import MapFrameIngestResponse, MapStreamStatus
from app.services.map_service import map_service

router = APIRouter(prefix="/map", tags=["map"])


_ONE_PX_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
    b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xcf \xff\xd9"
)


@router.get("/status", response_model=MapStreamStatus)
def get_map_status() -> MapStreamStatus:
    return map_service.get_status()


@router.get("/snapshot.jpg")
def get_snapshot_jpeg() -> Response:
    jpeg = map_service.get_latest_jpeg_bytes() or _ONE_PX_JPEG
    return Response(content=jpeg, media_type="image/jpeg")


def _mjpeg_stream(frame_interval_ms: int) -> Iterator[bytes]:
    sleep_s = max(frame_interval_ms, 1) / 1000.0
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    last_frame = _ONE_PX_JPEG

    while True:
        frame = map_service.get_latest_jpeg_bytes() or last_frame
        if frame:
            last_frame = frame
        yield boundary + last_frame + b"\r\n"
        time.sleep(sleep_s)


@router.get("/stream")
def get_stream(
    frame_interval_ms: int = Query(default=settings.map_stream_interval_ms, ge=50, le=2000),
) -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_stream(frame_interval_ms),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/frame", response_model=MapFrameIngestResponse)
async def post_map_frame(
    request: Request,
    width: Optional[int] = Query(default=None, ge=1),
    height: Optional[int] = Query(default=None, ge=1),
    source: Optional[str] = Query(default=None),
    timestamp_ms: Optional[int] = Query(default=None, ge=0),
) -> MapFrameIngestResponse:
    payload = await request.body()
    frame_seq = map_service.ingest_jpeg(
        payload,
        width=width,
        height=height,
        source=source,
        source_timestamp_ms=timestamp_ms,
    )
    return MapFrameIngestResponse(
        ok=True,
        timestamp_ms=int(time.time() * 1000),
        frame_seq=frame_seq,
    )
