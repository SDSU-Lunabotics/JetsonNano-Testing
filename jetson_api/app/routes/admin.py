from fastapi import APIRouter, HTTPException

from app.schemas.admin import RestartServiceRequest, AdminActionResponse
from app.services.admin_service import admin_service

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/restart-service", response_model=AdminActionResponse)
def restart_service(req: RestartServiceRequest) -> AdminActionResponse:
    try:
        return admin_service.restart_service(req.service_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/restart-jetson", response_model=AdminActionResponse)
def restart_jetson() -> AdminActionResponse:
    try:
        return admin_service.restart_jetson()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))