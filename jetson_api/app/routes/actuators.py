from fastapi import APIRouter, HTTPException

from app.services.motor_service import motor_service
from app.services.state_service import state_service, now_ms

router = APIRouter(prefix="/actuators", tags=["actuators"])

# if e-stop is on, prevent starting any actuators. Stopping should still work to allow clearing the e-stop condition.
def _guard():
    if state_service.estop:
        raise HTTPException(status_code=409, detail="E-stop active")


# placeholder endpoints for controlling actuators - not hardware implemented yet

@router.get("/status")
def actuator_status():
    motors_status = motor_service.get_status()
    excavation = motors_status.telemetry.excavation
    deposition = motors_status.telemetry.deposition
    return {
        "ok": True,
        "timestamp_ms": motors_status.timestamp_ms,
        "excavation": {
            "left_extension_inches": excavation.left_extension_inches,
            "right_extension_inches": excavation.right_extension_inches,
            "left_extension_pct": excavation.left_extension_pct,
            "right_extension_pct": excavation.right_extension_pct,
            "position_calibrated": excavation.position_calibrated,
            "sync_fault": excavation.sync_fault,
            "manual_bypass_sync": excavation.manual_bypass_sync,
            "state": excavation.state,
            "sources": {
                "jetson_left_extension_inches": excavation.jetson_left_extension_inches,
                "jetson_right_extension_inches": excavation.jetson_right_extension_inches,
                "excav_bottom_left_inches": excavation.bottom_left_inches,
                "excav_bottom_right_inches": excavation.bottom_right_inches,
            },
        },
        "tailgate": {
            "counts": deposition.tailgate_counts,
            "inches": deposition.tailgate_inches,
            "extension_pct": deposition.tailgate_extension_pct,
            "position_calibrated": deposition.tailgate_position_calibrated,
            "state": deposition.tailgate_state,
            "moving": deposition.tailgate_moving,
            "open": deposition.tailgate_open,
            "closed": deposition.tailgate_closed,
            "torque_current_a": deposition.tailgate_torque_current_a,
            "sources": {
                "deposit_tailgate_counts": motors_status.raw_values.get("Deposit/TailgateCounts"),
                "tailgate_counts": motors_status.raw_values.get("Tailgate/Counts"),
                "jetson_tailgate_counts": motors_status.raw_values.get("Jetson/TailgateCounts"),
            },
        },
    }


@router.post("/excavator/start")
def excavator_start():
    _guard()
    state_service.excavator_running = True
    return {"ok": True, "timestamp_ms": now_ms()}


@router.post("/excavator/stop")
def excavator_stop():
    state_service.excavator_running = False
    return {"ok": True, "timestamp_ms": now_ms()}


@router.post("/deposition/start")
def deposition_start():
    _guard()
    state_service.deposition_running = True
    return {"ok": True, "timestamp_ms": now_ms()}


@router.post("/deposition/stop")
def deposition_stop():
    state_service.deposition_running = False
    return {"ok": True, "timestamp_ms": now_ms()}


@router.post("/conveyor/start")
def conveyor_start():
    return deposition_start()


@router.post("/conveyor/stop")
def conveyor_stop():
    return deposition_stop()
