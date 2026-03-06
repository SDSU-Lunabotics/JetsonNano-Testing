import subprocess
from fastapi import APIRouter, HTTPException

from app.core.settings import settings
from app.schemas.admin import RestartServiceRequest, AdminActionResponse
from app.services.state_service import now_ms

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/restart-service", response_model=AdminActionResponse)
def restart_service(req: RestartServiceRequest) -> AdminActionResponse:
    if settings.dry_run:
        return AdminActionResponse(
            ok=True,
            message=f"Dry run: would restart service '{req.service_name}'",
            timestamp_ms=now_ms(),
        )

    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", req.service_name],
            check=True,
        )
        return AdminActionResponse(
            ok=True,
            message=f"Restarted service '{req.service_name}'",
            timestamp_ms=now_ms(),
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart-jetson", response_model=AdminActionResponse)
def restart_jetson() -> AdminActionResponse:
    if settings.dry_run or not settings.allow_reboot:
        return AdminActionResponse(
            ok=True,
            message="Dry run: would reboot Jetson",
            timestamp_ms=now_ms(),
        )

    try:
        subprocess.Popen(["sudo", "reboot"])
        return AdminActionResponse(
            ok=True,
            message="Reboot command issued",
            timestamp_ms=now_ms(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))