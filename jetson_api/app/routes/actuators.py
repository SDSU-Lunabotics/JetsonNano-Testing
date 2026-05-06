from fastapi import APIRouter, HTTPException

from app.services.state_service import state_service, now_ms
from app.services.motor_service import motor_service
from app.schemas.common import MotorId
from app.schemas.motors import MotorCommandRequest

router = APIRouter(prefix="/actuators", tags=["actuators"])

# if e-stop is on, prevent starting any actuators. Stopping should still work to allow clearing the e-stop condition.
def _guard():
    if state_service.estop:
        raise HTTPException(status_code=409, detail="E-stop active")


# placeholder endpoints for controlling actuators - not hardware implemented yet

@router.post("/excavator/start")
def excavator_start():
    _guard()
    state_service.excavator_running = True
    motor_service.command_motor(MotorId.excavator, MotorCommandRequest(mode="percent", value=1.0))
    return {"ok": True, "timestamp_ms": now_ms()}


@router.post("/excavator/stop")
def excavator_stop():
    state_service.excavator_running = False
    motor_service.command_motor(MotorId.excavator, MotorCommandRequest(mode="stop"))
    return {"ok": True, "timestamp_ms": now_ms()}


@router.post("/deposition/start")
def deposition_start():
    _guard()
    state_service.deposition_running = True
    motor_service.command_motor(MotorId.deposition, MotorCommandRequest(mode="percent", value=1.0))
    return {"ok": True, "timestamp_ms": now_ms()}


@router.post("/deposition/stop")
def deposition_stop():
    state_service.deposition_running = False
    motor_service.command_motor(MotorId.deposition, MotorCommandRequest(mode="stop"))
    return {"ok": True, "timestamp_ms": now_ms()}


@router.post("/conveyor/start")
def conveyor_start():
    return deposition_start()


@router.post("/conveyor/stop")
def conveyor_stop():
    return deposition_stop()
