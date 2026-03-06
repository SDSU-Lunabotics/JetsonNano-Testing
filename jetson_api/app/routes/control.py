from fastapi import APIRouter, HTTPException

from app.schemas.control import (
    LedRequest,
    LedResponse,
    DriveForwardRequest,
    DriveForwardResponse,
    EstopRequest,
    EstopResponse,
)
from app.services.nt_service import nt_service
from app.services.state_service import state_service, now_ms

router = APIRouter(tags=["control"])


@router.post("/led", response_model=LedResponse)
def set_led(req: LedRequest) -> LedResponse:
    if not nt_service.wait_for_connection():
        raise HTTPException(status_code=503, detail="RoboRIO not connected")

    nt_service.run_async(nt_service.set_led, req.state, req.override)
    state_service.led_state = req.state
    state_service.led_override = req.override

    return LedResponse(
        ok=True,
        state=req.state,
        override=req.override,
        timestamp_ms=now_ms(),
    )


@router.post("/drive/forward", response_model=DriveForwardResponse)
def drive_forward(req: DriveForwardRequest) -> DriveForwardResponse:
    if state_service.estop:
        raise HTTPException(status_code=409, detail="E-stop active")

    if not nt_service.wait_for_connection():
        raise HTTPException(status_code=503, detail="RoboRIO not connected")

    state_service.automation_enabled = True
    nt_service.run_async(nt_service.drive_forward, req.duration, req.speed)

    return DriveForwardResponse(
        ok=True,
        action="drive_forward",
        duration=req.duration,
        speed=req.speed,
        timestamp_ms=now_ms(),
    )


@router.post("/control/estop", response_model=EstopResponse)
def set_estop(req: EstopRequest) -> EstopResponse:
    state_service.estop = req.engage

    if req.engage:
        state_service.automation_enabled = False
        state_service.excavator_running = False
        state_service.conveyor_running = False

        if nt_service.is_connected():
            nt_service.run_async(nt_service.stop_all_motion)

    return EstopResponse(
        ok=True,
        estop=state_service.estop,
        timestamp_ms=now_ms(),
    )