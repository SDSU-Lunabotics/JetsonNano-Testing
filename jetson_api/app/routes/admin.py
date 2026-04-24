from fastapi import APIRouter, HTTPException

from app.schemas.admin import RestartServiceRequest, AdminActionResponse
from app.services.admin_service import admin_service

router = APIRouter(tags=["system"])


@router.post("/system/restart-service", response_model=AdminActionResponse)
@router.post("/admin/restart-service", response_model=AdminActionResponse, include_in_schema=False)
def restart_service(req: RestartServiceRequest) -> AdminActionResponse:
    try:
        return admin_service.restart_service(req.service_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/system/restart-jetson", response_model=AdminActionResponse)
@router.post("/admin/restart-jetson", response_model=AdminActionResponse, include_in_schema=False)
def restart_jetson() -> AdminActionResponse:
    try:
        return admin_service.restart_jetson()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
