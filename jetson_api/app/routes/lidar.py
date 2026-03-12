import asyncio
import time
from fastapi import APIRouter, Response, WebSocket, WebSocketDisconnect

from app.schemas.lidar import (
    LidarStatusResponse,
    LidarMapInfoResponse,
    LidarPreviewMessage,
)
from app.services.lidar_service import lidar_service

router = APIRouter(prefix="/lidar", tags=["lidar"])


def _now_ms() -> int:
    return int(time.time() * 1000)


_ONE_PX_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


@router.get("/status", response_model=LidarStatusResponse)
def get_lidar_status() -> LidarStatusResponse:
    return lidar_service.get_status()


@router.get("/map/info", response_model=LidarMapInfoResponse)
def get_map_info() -> LidarMapInfoResponse:
    return lidar_service.get_map_info()


@router.get("/map")
def get_map_image() -> Response:
    png = lidar_service.get_map_png() or _ONE_PX_PNG
    return Response(content=png, media_type="image/png")


@router.websocket("/ws")
async def ws_lidar(websocket: WebSocket) -> None:
    await websocket.accept()
    seq = 0
    try:
        while True:
            seq += 1
            msg = LidarPreviewMessage(
                type="lidar_preview",
                seq=seq,
                timestamp_ms=_now_ms(),
            )
            await websocket.send_json(msg.model_dump())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return