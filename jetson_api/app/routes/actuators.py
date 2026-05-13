from fastapi import APIRouter, HTTPException

from app.services.motor_service import motor_service
from app.services.state_service import state_service, now_ms

router = APIRouter(prefix="/actuators", tags=["actuators"])

# if e-stop is on, prevent starting any actuators. Stopping should still work to allow clearing the e-stop condition.
def _guard():
    if state_service.estop:
        raise HTTPException(status_code=409, detail="E-stop active")


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _source_key(raw_values, exact_keys, *, required=(), any_of=(), excluded=()):
    for key in exact_keys:
        if raw_values.get(key) is not None:
            return key

    for key, value in raw_values.items():
        if value is None:
            continue
        normalized = _normalize_key(key)
        if any(token in normalized for token in excluded):
            continue
        if not all(token in normalized for token in required):
            continue
        if any_of and not any(token in normalized for token in any_of):
            continue
        return key

    return None


def _source_value(raw_values, key):
    return None if key is None else raw_values.get(key)


@router.get("/status")
def actuator_status():
    motors_status = motor_service.get_status()
    excavation = motors_status.telemetry.excavation
    deposition = motors_status.telemetry.deposition
    raw_values = motors_status.raw_values
    left_pct_key = _source_key(
        raw_values,
        (
            "Excav/BotLeftExtensionPct",
            "Jetson/ExcavatorLeftExtensionPct",
            "Jetson/LeftActuatorExtensionPct",
            "Excavator/LeftExtensionPct",
        ),
        required=("left",),
        any_of=("pct", "percent"),
        excluded=("right", "tailgate", "gate"),
    )
    right_pct_key = _source_key(
        raw_values,
        (
            "Excav/BotRightExtensionPct",
            "Jetson/ExcavatorRightExtensionPct",
            "Jetson/RightActuatorExtensionPct",
            "Excavator/RightExtensionPct",
        ),
        required=("right",),
        any_of=("pct", "percent"),
        excluded=("left", "tailgate", "gate"),
    )
    left_inches_key = _source_key(
        raw_values,
        (
            "Excav/BotLeftInches",
            "Jetson/ExcavatorLeftExtensionInches",
            "Jetson/ExcavatorLeftInches",
            "Jetson/LeftActuatorInches",
            "Excavator/LeftInches",
            "Excavator/LeftExtensionInches",
        ),
        required=("left",),
        any_of=("inch",),
        excluded=("right", "tailgate", "gate"),
    )
    right_inches_key = _source_key(
        raw_values,
        (
            "Excav/BotRightInches",
            "Jetson/ExcavatorRightExtensionInches",
            "Jetson/ExcavatorRightInches",
            "Jetson/RightActuatorInches",
            "Excavator/RightInches",
            "Excavator/RightExtensionInches",
        ),
        required=("right",),
        any_of=("inch",),
        excluded=("left", "tailgate", "gate"),
    )
    tailgate_counts_key = _source_key(
        raw_values,
        ("Deposit/TailgateCounts", "Tailgate/Counts", "Jetson/TailgateCounts", "Jetson/GateActuatorCounts", "GateActuator/Counts"),
        required=("tailgate",),
        any_of=("count",),
    )
    tailgate_inches_key = _source_key(
        raw_values,
        ("Deposit/TailgateInches", "Tailgate/Inches", "Jetson/TailgateInches", "Jetson/GateActuatorInches", "GateActuator/Inches"),
        required=("tailgate",),
        any_of=("inch",),
    )
    tailgate_pct_key = _source_key(
        raw_values,
        ("Deposit/TailgateExtensionPct", "Tailgate/ExtensionPct", "Jetson/TailgateExtensionPct", "Jetson/GateActuatorExtensionPct", "GateActuator/ExtensionPct"),
        required=("tailgate",),
        any_of=("pct", "percent"),
    )
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
                "left_extension_pct": left_pct_key,
                "left_extension_pct_raw": _source_value(raw_values, left_pct_key),
                "right_extension_pct": right_pct_key,
                "right_extension_pct_raw": _source_value(raw_values, right_pct_key),
                "left_extension_inches": left_inches_key,
                "left_extension_inches_raw": _source_value(raw_values, left_inches_key),
                "right_extension_inches": right_inches_key,
                "right_extension_inches_raw": _source_value(raw_values, right_inches_key),
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
            "direction": deposition.tailgate_direction,
            "moving": deposition.tailgate_moving,
            "open": deposition.tailgate_open,
            "closed": deposition.tailgate_closed,
            "torque_current_a": deposition.tailgate_torque_current_a,
            "sources": {
                "counts": tailgate_counts_key,
                "counts_raw": _source_value(raw_values, tailgate_counts_key),
                "inches": tailgate_inches_key,
                "inches_raw": _source_value(raw_values, tailgate_inches_key),
                "extension_pct": tailgate_pct_key,
                "extension_pct_raw": _source_value(raw_values, tailgate_pct_key),
            },
        },
        "raw_values": raw_values,
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
