import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.settings import settings
from app.schemas.control import (
    LedRequest,
    LedResponse,
    DriveForwardRequest,
    DriveForwardResponse,
    EstopRequest,
    EstopResponse,
    ControlModeRequest,
    ControlModeResponse,
)
from app.services.roborio_bridge_service import roborio_bridge_service
from app.services.nt_service import nt_service
from app.services.state_service import state_service, now_ms

router = APIRouter(tags=["control"])


def _write_zed_command(command: dict) -> None:
    target = Path(settings.map_waypoint_command_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f"{target.suffix}.tmp")
    tmp.write_text(json.dumps(command), encoding="utf-8")
    tmp.replace(target)


def _run_manual_drive(duration: float, speed: float) -> None:
    """
    Wrapper around the NT drive command so we can keep Jetson-side state in sync.
    """
    try:
        nt_service.drive_forward(duration, speed)
    finally:
        # Clear manual-motion state when the command finishes
        state_service.manual_motion_active = False


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

    # This is a manual command, not true autonomy
    state_service.manual_motion_active = True
    state_service.last_drive_speed = req.speed
    state_service.last_drive_duration = req.duration
    state_service.last_drive_timestamp_ms = now_ms()

    nt_service.run_async(_run_manual_drive, req.duration, req.speed)

    return DriveForwardResponse(
        ok=True,
        action="drive_forward",
        duration=req.duration,
        speed=req.speed,
        command_source="manual",
        timestamp_ms=now_ms(),
    )


@router.post("/control/estop", response_model=EstopResponse)
def set_estop(req: EstopRequest) -> EstopResponse:
    if req.engage:
        state_service.estop = True
    else:
        state_service.estop = False

    try:
        roborio_bridge_service.set_estop(req.engage)
    except ValueError as exc:
        if not req.engage:
            # Stay latched on the Jetson side if we could not safely clear the RoboRIO e-stop.
            state_service.estop = True
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if req.engage:
        state_service.autonomy_enabled = False
        state_service.manual_motion_active = False
        state_service.excavator_running = False
        state_service.deposition_running = False

        # Clear last drive command info on estop
        state_service.last_drive_speed = None
        state_service.last_drive_duration = None
        state_service.last_drive_timestamp_ms = None

        if nt_service.is_connected():
            nt_service.run_async(nt_service.stop_all_motion)

    return EstopResponse(
        ok=True,
        estop=state_service.estop,
        timestamp_ms=now_ms(),
    )


@router.post("/control/mode", response_model=ControlModeResponse)
def set_control_mode(req: ControlModeRequest) -> ControlModeResponse:
    if state_service.estop and req.mode == "autonomy":
        raise HTTPException(status_code=409, detail="E-stop active")

    command_seq = now_ms()
    try:
        _write_zed_command(
            {
                "seq": command_seq,
                "type": "ui_action",
                "action": "set_control_mode",
                "mode": req.mode,
                "source": "jetson_api",
                "timestamp_ms": command_seq,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue mode command: {exc}") from exc

    state_service.autonomy_enabled = req.mode == "autonomy"
    if req.mode == "manual":
        state_service.manual_motion_active = False

    return ControlModeResponse(
        ok=True,
        mode=req.mode,
        autonomy_running=state_service.autonomy_enabled,
        command_seq=command_seq,
        timestamp_ms=command_seq,
    )
