from fastapi import APIRouter
from app.schemas.status import JetsonStatusResponse
from app.services.state_service import state_service

router = APIRouter(tags=["status"])


@router.get("/status", response_model=JetsonStatusResponse)
def get_status() -> JetsonStatusResponse:
    return JetsonStatusResponse(**state_service.status_dict())