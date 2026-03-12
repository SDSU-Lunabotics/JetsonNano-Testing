from fastapi import APIRouter, Response

from app.schemas.camera import CameraModeRequest, CameraModeResponse
from app.services.camera_service import camera_service

router = APIRouter(prefix="/camera", tags=["camera"])


_ONE_PX_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


@router.get("/snapshot")
def get_snapshot() -> Response:
    png = camera_service.get_snapshot_bytes() or _ONE_PX_PNG
    return Response(content=png, media_type="image/png")


@router.post("/mode", response_model=CameraModeResponse)
def set_camera_mode(req: CameraModeRequest) -> CameraModeResponse:
    return camera_service.set_mode(req)