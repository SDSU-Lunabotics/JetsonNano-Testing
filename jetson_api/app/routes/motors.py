from fastapi import APIRouter, HTTPException

from app.schemas.motors import MotorsStatusResponse, MotorCommandRequest, MotorCommandResponse
from app.services.motor_service import motor_service

router = APIRouter(prefix="/motors", tags=["motors"])


@router.get("/status", response_model=MotorsStatusResponse)
def get_motors_status() -> MotorsStatusResponse:
    return motor_service.get_status()


@router.post("/{motor_id}/command", response_model=MotorCommandResponse)
def motor_command(motor_id: str, req: MotorCommandRequest) -> MotorCommandResponse:
    try:
        resolved_motor_id = motor_service.resolve_motor_id(motor_id)
        return motor_service.command_motor(resolved_motor_id, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
