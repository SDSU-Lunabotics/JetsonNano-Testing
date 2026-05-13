from typing import Any, Dict, List, Optional, Tuple

from app.schemas.common import Fault, FaultCode, MotorId
from app.schemas.motors import (
    CurrentLimitsTelemetry,
    DepositionTelemetry,
    DriveMotorTelemetry,
    DriveTelemetry,
    ExcavationTargetsTelemetry,
    ExcavationTelemetry,
    JetsonMotorTelemetry,
    MotorCardsTelemetry,
    MotorCardStatus,
    MotorHealth,
    MotorWarning,
    MotorsStatusResponse,
    MotorCommandRequest,
    MotorCommandResponse,
)
from app.services.state_service import now_ms
from app.services.roborio_bridge_service import roborio_bridge_service


class MotorService:
    def get_status(self) -> MotorsStatusResponse:
        payload = roborio_bridge_service.get_motors_status()
        timestamp = int(payload.get("timestamp_ms") or now_ms())
        raw_values = dict(payload.get("values") or {})
        telemetry = _build_jetson_motor_telemetry(raw_values)
        motors = _build_motor_health_rows(raw_values, telemetry, timestamp)

        for raw_motor in payload.get("motors", []):
            normalized = _normalize_motor_payload(raw_motor)
            motor_id_raw = normalized.get("motor_id")
            if motor_id_raw is None:
                continue

            try:
                motor_id = _parse_motor_id(motor_id_raw)
            except ValueError:
                continue

            faults: List[Fault] = []
            if normalized.get("fault"):
                bridge_fault_code = normalized.get("fault_code")
                message = bridge_fault_code or f"{motor_id.value} reported a fault"
                faults.append(
                    Fault(
                        code=FaultCode.MOTOR_TORQUE_FAULT,
                        severity="error",
                        message=message,
                        source="roborio",
                        timestamp_ms=timestamp,
                    )
                )

            motors.append(
                MotorHealth(
                    motor_id=motor_id,
                    connected=_optional_bool(normalized.get("connected")),
                    enabled=_optional_bool(normalized.get("enabled")),
                    active=_optional_bool(normalized.get("active")),
                    healthy=_optional_bool(normalized.get("healthy")),
                    current_a=_optional_float(normalized.get("current_a")),
                    rpm=_optional_float(normalized.get("rpm")),
                    torque_nm=_optional_float(normalized.get("torque_nm")),
                    voltage_v=_optional_float(normalized.get("voltage_v")),
                    temperature_c=_optional_float(normalized.get("temperature_c")),
                    position_rotations=_optional_float(normalized.get("position_rotations")),
                    output_percent=_optional_float(normalized.get("output_percent")),
                    target_rpm=_optional_float(normalized.get("target_rpm")),
                    target_percent=_optional_float(normalized.get("target_percent")),
                    controller_mode=_optional_str(normalized.get("controller_mode")),
                    state=_optional_str(normalized.get("state")),
                    fault=_optional_bool(normalized.get("fault")),
                    fault_code=_optional_str(normalized.get("fault_code")),
                    timestamp_ms=_optional_int(normalized.get("timestamp_ms")),
                    last_update_ms=_optional_int(normalized.get("last_update_ms")) or timestamp,
                    faults=faults,
                    raw=_raw_passthrough(normalized),
                )
            )

        return MotorsStatusResponse(
            timestamp_ms=timestamp,
            motors=motors,
            telemetry=telemetry,
            cards=_build_motor_cards(telemetry),
            raw_values=raw_values,
            warnings=_parse_motor_warnings(payload.get("warnings")),
        )

    def command_motor(self, motor_id: MotorId, req: MotorCommandRequest) -> MotorCommandResponse:
        response = roborio_bridge_service.command_motor(
            motor_id=motor_id.value,
            mode=req.mode,
            value=req.value,
            duration_ms=req.duration_ms,
        )
        return MotorCommandResponse(
            ok=bool(response.get("ok", False)),
            motor_id=str(response.get("motor_id", motor_id.value)),
            mode=str(response.get("mode", req.mode)),
            value=_optional_float(response.get("value")),
            duration_ms=_optional_int(response.get("duration_ms")),
            request_id=_optional_float(response.get("request_id")),
            timestamp_ms=int(response.get("timestamp_ms") or now_ms()),
        )

    def resolve_motor_id(self, motor_id: Any) -> MotorId:
        return _parse_motor_id(motor_id)


def _optional_float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None


def _optional_int(value):
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_str(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_float(values: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _optional_float(values.get(key))
        if value is not None:
            return value
    return None


def _first_bool(values: Dict[str, Any], *keys: str) -> Optional[bool]:
    for key in keys:
        value = _optional_bool(values.get(key))
        if value is not None:
            return value
    return None


def _first_str(values: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = _optional_str(values.get(key))
        if value is not None:
            return value
    return None


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _dynamic_key(
    values: Dict[str, Any],
    *,
    required: Tuple[str, ...],
    any_of: Tuple[str, ...] = (),
    excluded: Tuple[str, ...] = (),
) -> Optional[str]:
    for key, value in values.items():
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


def _first_float_dynamic(
    values: Dict[str, Any],
    keys: Tuple[str, ...],
    *,
    required: Tuple[str, ...],
    any_of: Tuple[str, ...] = (),
    excluded: Tuple[str, ...] = (),
) -> Optional[float]:
    exact = _first_float(values, *keys)
    if exact is not None:
        return exact
    dynamic = _dynamic_key(values, required=required, any_of=any_of, excluded=excluded)
    return None if dynamic is None else _optional_float(values.get(dynamic))


def _first_bool_dynamic(
    values: Dict[str, Any],
    keys: Tuple[str, ...],
    *,
    required: Tuple[str, ...],
    any_of: Tuple[str, ...] = (),
    excluded: Tuple[str, ...] = (),
) -> Optional[bool]:
    exact = _first_bool(values, *keys)
    if exact is not None:
        return exact
    dynamic = _dynamic_key(values, required=required, any_of=any_of, excluded=excluded)
    return None if dynamic is None else _optional_bool(values.get(dynamic))


def _first_str_dynamic(
    values: Dict[str, Any],
    keys: Tuple[str, ...],
    *,
    required: Tuple[str, ...],
    any_of: Tuple[str, ...] = (),
    excluded: Tuple[str, ...] = (),
) -> Optional[str]:
    exact = _first_str(values, *keys)
    if exact is not None:
        return exact
    dynamic = _dynamic_key(values, required=required, any_of=any_of, excluded=excluded)
    return None if dynamic is None else _optional_str(values.get(dynamic))


def _parse_motor_id(value: Any) -> MotorId:
    raw = str(value).strip()
    candidates = [
        raw,
        raw.lower(),
        raw.replace("-", "_").replace(" ", "_").lower(),
        _camel_to_snake(raw),
    ]

    alias_map = {
        "front_left": MotorId.left_front,
        "left_front_drive": MotorId.left_front,
        "drive_left_front": MotorId.left_front,
        "front_right": MotorId.right_front,
        "right_front_drive": MotorId.right_front,
        "drive_right_front": MotorId.right_front,
        "rear_left": MotorId.left_rear,
        "left_rear_drive": MotorId.left_rear,
        "drive_left_rear": MotorId.left_rear,
        "rear_right": MotorId.right_rear,
        "right_rear_drive": MotorId.right_rear,
        "drive_right_rear": MotorId.right_rear,
        "conveyor": MotorId.deposition,
    }

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return MotorId(candidate)
        except ValueError:
            mapped = alias_map.get(candidate)
            if mapped is not None:
                return mapped

    raise ValueError(f"Unsupported motor id: {value}")


def _camel_to_snake(value: str) -> str:
    chars: List[str] = []
    for index, ch in enumerate(value.strip()):
        if ch.isupper() and index > 0 and value[index - 1].isalnum():
            chars.append("_")
        chars.append(ch.lower())
    return "".join(chars).replace("-", "_").replace(" ", "_")


def _normalize_motor_payload(raw_motor: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(raw_motor or {})

    alias_pairs = {
        "motor_id": ["motor_id", "motorId", "id", "name"],
        "connected": ["connected", "is_connected", "online"],
        "enabled": ["enabled", "is_enabled", "active"],
        "healthy": ["healthy", "ok", "nominal"],
        "fault": ["fault", "has_fault", "faulted"],
        "fault_code": ["fault_code", "faultCode", "fault_reason", "faultReason"],
        "current_a": ["current_a", "current", "currentAmps", "amps", "supply_current_a"],
        "rpm": ["rpm", "velocity_rpm", "speed_rpm"],
        "torque_nm": ["torque_nm", "torque", "torqueNm"],
        "voltage_v": ["voltage_v", "voltage", "bus_voltage_v", "busVoltage"],
        "temperature_c": ["temperature_c", "temperature", "temp_c", "tempC"],
        "position_rotations": ["position_rotations", "position", "rotations", "encoder_rotations"],
        "output_percent": ["output_percent", "output", "applied_output", "percent_output"],
        "target_rpm": ["target_rpm", "setpoint_rpm", "commanded_rpm"],
        "target_percent": ["target_percent", "setpoint_percent", "commanded_percent"],
        "controller_mode": ["controller_mode", "control_mode", "mode"],
        "timestamp_ms": ["timestamp_ms", "timestamp", "sample_timestamp_ms"],
        "last_update_ms": ["last_update_ms", "updated_at_ms", "lastUpdateMs"],
    }

    for canonical, aliases in alias_pairs.items():
        for alias in aliases:
            if alias in raw_motor and raw_motor.get(alias) is not None:
                normalized[canonical] = raw_motor.get(alias)
                break

    return normalized


def _raw_passthrough(normalized: Dict[str, Any]) -> Dict[str, Any]:
    typed_keys = {
        "motor_id",
        "connected",
        "enabled",
        "active",
        "healthy",
        "fault",
        "fault_code",
        "current_a",
        "rpm",
        "torque_nm",
        "voltage_v",
        "temperature_c",
        "position_rotations",
        "output_percent",
        "target_rpm",
        "target_percent",
        "controller_mode",
        "state",
        "timestamp_ms",
        "last_update_ms",
    }
    return {key: value for key, value in normalized.items() if key not in typed_keys}


def _warning_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_motor_warnings(raw_warnings: Any) -> List[MotorWarning]:
    warnings: List[MotorWarning] = []
    for item in raw_warnings or []:
        if not isinstance(item, dict):
            continue
        key = _optional_str(item.get("key"))
        if not key:
            continue
        warnings.append(
            MotorWarning(
                key=key,
                value=_warning_float(item.get("value")),
                limit=_warning_float(item.get("limit")),
            )
        )
    return warnings


def _build_jetson_motor_telemetry(values: Dict[str, Any]) -> JetsonMotorTelemetry:
    left_extension_inches = _first_float_dynamic(
        values,
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

    right_extension_inches = _first_float_dynamic(
        values,
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

    left_extension_pct = _first_float_dynamic(
        values,
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

    right_extension_pct = _first_float_dynamic(
        values,
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

    position_calibrated = _first_bool_dynamic(
        values,
        (
            "Excav/BottomPositionCalibrated",
            "Jetson/ExcavatorBottomPositionCalibrated",
        ),
        required=("calibrated",),
        any_of=("excav", "actuator", "bottom"),
        excluded=("tailgate", "gate"),
    )

    return JetsonMotorTelemetry(
        drive=DriveTelemetry(
            left_front=_drive_motor(values, "Kraken/LeftFront"),
            left_rear=_drive_motor(values, "Kraken/LeftRear"),
            right_front=_drive_motor(values, "Kraken/RightFront"),
            right_rear=_drive_motor(values, "Kraken/RightRear"),
            left_output=_optional_float(values.get("Jetson/DriveLeftOutput")),
            right_output=_optional_float(values.get("Jetson/DriveRightOutput")),
            active=_optional_bool(values.get("Jetson/DriveActive")),
            wheel_test_step=_optional_float(values.get("Kraken/WheelTestStep")),
            wheel_test_mode=_optional_str(values.get("Kraken/WheelTestMode")),
        ),
        excavation=ExcavationTelemetry(
            belt_running=_optional_bool(values.get("Excav/BeltRunning")),
            belt_output=_optional_float(values.get("Excav/BeltOutput")),
            belt_leader_torque_current_a=_optional_float(values.get("Excav/BeltLeaderTorqueCurrentA")),
            belt_follower_torque_current_a=_optional_float(values.get("Excav/BeltFollowerTorqueCurrentA")),
            left_extension_inches=left_extension_inches,
            right_extension_inches=right_extension_inches,
            left_extension_pct=left_extension_pct,
            right_extension_pct=right_extension_pct,
            position_calibrated=position_calibrated,
            bottom_left_counts=_first_float_dynamic(
                values,
                ("Excav/BotLeftCounts", "Jetson/ExcavatorLeftCounts", "Jetson/LeftActuatorCounts"),
                required=("left",),
                any_of=("count",),
                excluded=("right", "tailgate", "gate"),
            ),
            bottom_right_counts=_first_float_dynamic(
                values,
                ("Excav/BotRightCounts", "Jetson/ExcavatorRightCounts", "Jetson/RightActuatorCounts"),
                required=("right",),
                any_of=("count",),
                excluded=("left", "tailgate", "gate"),
            ),
            bottom_left_inches=left_extension_inches,
            bottom_right_inches=right_extension_inches,
            bottom_left_extension_pct=left_extension_pct,
            bottom_right_extension_pct=right_extension_pct,
            jetson_left_extension_inches=_optional_float(values.get("Jetson/ExcavatorLeftExtensionInches")),
            jetson_right_extension_inches=_optional_float(values.get("Jetson/ExcavatorRightExtensionInches")),
            jetson_left_extension_pct=_optional_float(values.get("Jetson/ExcavatorLeftExtensionPct")),
            jetson_right_extension_pct=_optional_float(values.get("Jetson/ExcavatorRightExtensionPct")),
            jetson_bottom_position_calibrated=_optional_bool(values.get("Jetson/ExcavatorBottomPositionCalibrated")),
            bottom_diff_counts=_optional_float(values.get("Excav/BottomDiffCounts")),
            sync_fault=_optional_bool(values.get("Excav/SyncFault")),
            manual_bypass_sync=_optional_bool(values.get("Excav/ManualBypassSync")),
            bottom_position_calibrated=_optional_bool(values.get("Excav/BottomPositionCalibrated")),
            position_reference=_optional_str(values.get("Excav/PositionReference")),
            state=_optional_str(values.get("Excav/State")),
            targets=ExcavationTargetsTelemetry(
                stow_bottom_counts=_optional_float(values.get("Excav/Targets/StowBottomCounts")),
                dig_zone_bottom_counts=_optional_float(values.get("Excav/Targets/DigZoneBottomCounts")),
            ),
        ),
        deposition=DepositionTelemetry(
            running=_optional_bool(values.get("Deposit/Running")),
            output=_optional_float(values.get("Deposit/Output")),
            collecting_assist=_optional_bool(values.get("Deposit/CollectingAssist")),
            depositing=_optional_bool(values.get("Deposit/Depositing")),
            door_state=_optional_str(values.get("Deposit/DoorState")),
            torque_current_a=_optional_float(values.get("Deposit/TorqueCurrentA")),
            tailgate_counts=_first_float_dynamic(
                values,
                (
                    "Deposit/TailgateCounts",
                    "Tailgate/Counts",
                    "Jetson/TailgateCounts",
                    "Jetson/GateActuatorCounts",
                    "GateActuator/Counts",
                ),
                required=("count",),
                any_of=("tailgate", "gateactuator"),
                excluded=("excav", "left", "right"),
            ),
            tailgate_inches=_first_float_dynamic(
                values,
                (
                    "Deposit/TailgateInches",
                    "Tailgate/Inches",
                    "Jetson/TailgateInches",
                    "Jetson/GateActuatorInches",
                    "GateActuator/Inches",
                ),
                required=("inch",),
                any_of=("tailgate", "gateactuator"),
                excluded=("excav", "left", "right", "pct", "percent", "count"),
            ),
            tailgate_extension_pct=_first_float_dynamic(
                values,
                (
                    "Deposit/TailgateExtensionPct",
                    "Tailgate/ExtensionPct",
                    "Jetson/TailgateExtensionPct",
                    "Jetson/GateActuatorExtensionPct",
                    "GateActuator/ExtensionPct",
                ),
                required=("extension",),
                any_of=("tailgate", "gateactuator"),
                excluded=("excav", "left", "right", "inch", "count"),
            ),
            tailgate_position_calibrated=_first_bool_dynamic(
                values,
                (
                    "Deposit/TailgatePositionCalibrated",
                    "Tailgate/PositionCalibrated",
                    "Jetson/TailgatePositionCalibrated",
                    "Jetson/GateActuatorPositionCalibrated",
                    "GateActuator/PositionCalibrated",
                ),
                required=("calibrated",),
                any_of=("tailgate", "gateactuator"),
                excluded=("excav", "left", "right"),
            ),
            tailgate_state=_first_str_dynamic(
                values,
                (
                    "Deposit/TailgateState",
                    "Tailgate/State",
                    "Jetson/TailgateState",
                    "Jetson/GateActuatorState",
                    "GateActuator/State",
                ),
                required=("state",),
                any_of=("tailgate", "gateactuator"),
                excluded=("excav", "left", "right"),
            ),
            tailgate_direction=_first_str_dynamic(
                values,
                (
                    "Deposit/TailgateDirection",
                    "Tailgate/Direction",
                    "Jetson/TailgateDirection",
                    "Jetson/GateActuatorDirection",
                    "GateActuator/Direction",
                ),
                required=("direction",),
                any_of=("tailgate", "gateactuator"),
                excluded=("excav", "left", "right"),
            ),
            tailgate_moving=_first_bool_dynamic(
                values,
                (
                    "Deposit/TailgateMoving",
                    "Tailgate/Moving",
                    "Jetson/TailgateMoving",
                    "GateActuator/Moving",
                ),
                required=("moving",),
                any_of=("tailgate", "gateactuator"),
                excluded=("excav", "left", "right"),
            ),
            tailgate_open=_first_bool_dynamic(
                values,
                (
                    "Deposit/TailgateOpen",
                    "Tailgate/Open",
                    "Jetson/TailgateOpen",
                    "GateActuator/Open",
                ),
                required=("open",),
                any_of=("tailgate", "gateactuator"),
                excluded=("excav", "left", "right"),
            ),
            tailgate_closed=_first_bool_dynamic(
                values,
                (
                    "Deposit/TailgateClosed",
                    "Tailgate/Closed",
                    "Jetson/TailgateClosed",
                    "GateActuator/Closed",
                ),
                required=("closed",),
                any_of=("tailgate", "gateactuator"),
                excluded=("excav", "left", "right"),
            ),
            tailgate_torque_current_a=_first_float_dynamic(
                values,
                (
                    "Deposit/TailgateTorqueCurrentA",
                    "Tailgate/TorqueCurrentA",
                    "Jetson/TailgateTorqueCurrentA",
                    "GateActuator/TorqueCurrentA",
                ),
                required=("current",),
                any_of=("tailgate", "gateactuator"),
                excluded=("excav", "left", "right"),
            ),
        ),
        current_limits=CurrentLimitsTelemetry(
            enabled=_optional_bool(values.get("MainRover/CurrentLimitEnabled")),
            drive_a=_optional_float(values.get("MainRover/DriveCurrentLimitA")),
            excav_a=_optional_float(values.get("MainRover/ExcavCurrentLimitA")),
            deposit_a=_optional_float(values.get("MainRover/DepositCurrentLimitA")),
            applied=_optional_bool(values.get("MainRover/CurrentLimitApplied")),
            applied_drive_a=_optional_float(values.get("MainRover/DriveCurrentLimitAppliedA")),
            applied_excav_a=_optional_float(values.get("MainRover/ExcavCurrentLimitAppliedA")),
            applied_deposit_a=_optional_float(values.get("MainRover/DepositCurrentLimitAppliedA")),
        ),
    )


def _build_motor_health_rows(
    values: Dict[str, Any],
    telemetry: JetsonMotorTelemetry,
    timestamp: int,
) -> List[MotorHealth]:
    return [
        MotorHealth(
            motor_id=MotorId.left_front,
            connected=True if _has_any(values, "Kraken/LeftFront/VelocityTps", "Kraken/LeftFront/TorqueCurrentA", "Kraken/LeftFront/Enabled", "Kraken/LeftFront/AppliedOutput") else None,
            enabled=telemetry.drive.left_front.enabled,
            active=_wheel_is_active(telemetry.drive.left_front),
            current_a=telemetry.drive.left_front.torque_current_a,
            output_percent=telemetry.drive.left_front.applied_output,
            last_update_ms=timestamp,
            raw={
                "velocity_tps": telemetry.drive.left_front.velocity_tps,
                "drive_side_output": telemetry.drive.left_output,
            },
        ),
        MotorHealth(
            motor_id=MotorId.left_rear,
            connected=True if _has_any(values, "Kraken/LeftRear/VelocityTps", "Kraken/LeftRear/TorqueCurrentA", "Kraken/LeftRear/Enabled", "Kraken/LeftRear/AppliedOutput") else None,
            enabled=telemetry.drive.left_rear.enabled,
            active=_wheel_is_active(telemetry.drive.left_rear),
            current_a=telemetry.drive.left_rear.torque_current_a,
            output_percent=telemetry.drive.left_rear.applied_output,
            last_update_ms=timestamp,
            raw={
                "velocity_tps": telemetry.drive.left_rear.velocity_tps,
                "drive_side_output": telemetry.drive.left_output,
            },
        ),
        MotorHealth(
            motor_id=MotorId.right_front,
            connected=True if _has_any(values, "Kraken/RightFront/VelocityTps", "Kraken/RightFront/TorqueCurrentA", "Kraken/RightFront/Enabled", "Kraken/RightFront/AppliedOutput") else None,
            enabled=telemetry.drive.right_front.enabled,
            active=_wheel_is_active(telemetry.drive.right_front),
            current_a=telemetry.drive.right_front.torque_current_a,
            output_percent=telemetry.drive.right_front.applied_output,
            last_update_ms=timestamp,
            raw={
                "velocity_tps": telemetry.drive.right_front.velocity_tps,
                "drive_side_output": telemetry.drive.right_output,
            },
        ),
        MotorHealth(
            motor_id=MotorId.right_rear,
            connected=True if _has_any(values, "Kraken/RightRear/VelocityTps", "Kraken/RightRear/TorqueCurrentA", "Kraken/RightRear/Enabled", "Kraken/RightRear/AppliedOutput") else None,
            enabled=telemetry.drive.right_rear.enabled,
            active=_wheel_is_active(telemetry.drive.right_rear),
            current_a=telemetry.drive.right_rear.torque_current_a,
            output_percent=telemetry.drive.right_rear.applied_output,
            last_update_ms=timestamp,
            raw={
                "velocity_tps": telemetry.drive.right_rear.velocity_tps,
                "drive_side_output": telemetry.drive.right_output,
            },
        ),
        MotorHealth(
            motor_id=MotorId.excavator,
            connected=True if _has_any(
                values,
                "Excav/BeltRunning",
                "Excav/BeltOutput",
                "Excav/BeltLeaderTorqueCurrentA",
                "Excav/BeltFollowerTorqueCurrentA",
                "Excav/State",
                "Jetson/ExcavatorLeftExtensionInches",
                "Jetson/ExcavatorRightExtensionInches",
            ) else None,
            enabled=_excavation_is_active(telemetry),
            active=_excavation_is_active(telemetry),
            current_a=_max_optional_float(
                telemetry.excavation.belt_leader_torque_current_a,
                telemetry.excavation.belt_follower_torque_current_a,
            ),
            output_percent=telemetry.excavation.belt_output,
            state=telemetry.excavation.state,
            last_update_ms=timestamp,
            raw={
                "belt_leader_torque_current_a": telemetry.excavation.belt_leader_torque_current_a,
                "belt_follower_torque_current_a": telemetry.excavation.belt_follower_torque_current_a,
                "left_extension_inches": telemetry.excavation.left_extension_inches,
                "right_extension_inches": telemetry.excavation.right_extension_inches,
                "left_extension_pct": telemetry.excavation.left_extension_pct,
                "right_extension_pct": telemetry.excavation.right_extension_pct,
                "position_calibrated": telemetry.excavation.position_calibrated,
                "bottom_left_counts": telemetry.excavation.bottom_left_counts,
                "bottom_right_counts": telemetry.excavation.bottom_right_counts,
                "bottom_left_inches": telemetry.excavation.bottom_left_inches,
                "bottom_right_inches": telemetry.excavation.bottom_right_inches,
                "bottom_left_extension_pct": telemetry.excavation.bottom_left_extension_pct,
                "bottom_right_extension_pct": telemetry.excavation.bottom_right_extension_pct,
                "jetson_left_extension_inches": telemetry.excavation.jetson_left_extension_inches,
                "jetson_right_extension_inches": telemetry.excavation.jetson_right_extension_inches,
                "jetson_left_extension_pct": telemetry.excavation.jetson_left_extension_pct,
                "jetson_right_extension_pct": telemetry.excavation.jetson_right_extension_pct,
                "jetson_bottom_position_calibrated": telemetry.excavation.jetson_bottom_position_calibrated,
                "bottom_diff_counts": telemetry.excavation.bottom_diff_counts,
                "position_reference": telemetry.excavation.position_reference,
                "bottom_position_calibrated": telemetry.excavation.bottom_position_calibrated,
                "sync_fault": telemetry.excavation.sync_fault,
                "manual_bypass_sync": telemetry.excavation.manual_bypass_sync,
            },
        ),
        MotorHealth(
            motor_id=MotorId.deposition,
            connected=True if _has_any(
                values,
                "Deposit/Running",
                "Deposit/Output",
                "Deposit/TorqueCurrentA",
                "Deposit/DoorState",
                "Deposit/TailgateCounts",
                "Deposit/TailgateInches",
                "Deposit/TailgateExtensionPct",
                "Tailgate/Counts",
                "Tailgate/Inches",
                "Tailgate/ExtensionPct",
                "Jetson/TailgateCounts",
                "Jetson/TailgateInches",
                "Jetson/TailgateExtensionPct",
            ) else None,
            enabled=_deposition_is_active(telemetry),
            active=_deposition_is_active(telemetry),
            current_a=_max_optional_float(
                telemetry.deposition.torque_current_a,
                telemetry.deposition.tailgate_torque_current_a,
            ),
            output_percent=telemetry.deposition.output,
            state=_compose_deposition_state(telemetry),
            last_update_ms=timestamp,
            raw={
                "door_state": telemetry.deposition.door_state,
                "collecting_assist": telemetry.deposition.collecting_assist,
                "depositing": telemetry.deposition.depositing,
                "tailgate_counts": telemetry.deposition.tailgate_counts,
                "tailgate_inches": telemetry.deposition.tailgate_inches,
                "tailgate_extension_pct": telemetry.deposition.tailgate_extension_pct,
                "tailgate_position_calibrated": telemetry.deposition.tailgate_position_calibrated,
                "tailgate_state": telemetry.deposition.tailgate_state,
                "tailgate_direction": telemetry.deposition.tailgate_direction,
                "tailgate_moving": telemetry.deposition.tailgate_moving,
                "tailgate_open": telemetry.deposition.tailgate_open,
                "tailgate_closed": telemetry.deposition.tailgate_closed,
                "tailgate_torque_current_a": telemetry.deposition.tailgate_torque_current_a,
            },
        ),
    ]


def _build_motor_cards(telemetry: JetsonMotorTelemetry) -> MotorCardsTelemetry:
    drive_card = MotorCardStatus(
        motor_id=MotorId.left_front,
        label="Drive",
        active=_drive_is_active(telemetry),
        enabled=_drive_is_enabled(telemetry),
        current_a=_sum_optional_floats(
            telemetry.drive.left_front.torque_current_a,
            telemetry.drive.left_rear.torque_current_a,
            telemetry.drive.right_front.torque_current_a,
            telemetry.drive.right_rear.torque_current_a,
        ),
        output_percent=_max_abs_optional_float(
            telemetry.drive.left_output,
            telemetry.drive.right_output,
        ),
        state=_drive_state_text(telemetry),
        source="Jetson/DriveActive",
    )
    left_front_card = MotorCardStatus(
        motor_id=MotorId.left_front,
        label="Left Front Drive",
        active=_wheel_is_active(telemetry.drive.left_front),
        enabled=telemetry.drive.left_front.enabled,
        current_a=telemetry.drive.left_front.torque_current_a,
        output_percent=telemetry.drive.left_front.applied_output,
        state=_wheel_state_text(telemetry.drive.left_front),
        source="Kraken/LeftFront/AppliedOutput",
    )
    left_rear_card = MotorCardStatus(
        motor_id=MotorId.left_rear,
        label="Left Rear Drive",
        active=_wheel_is_active(telemetry.drive.left_rear),
        enabled=telemetry.drive.left_rear.enabled,
        current_a=telemetry.drive.left_rear.torque_current_a,
        output_percent=telemetry.drive.left_rear.applied_output,
        state=_wheel_state_text(telemetry.drive.left_rear),
        source="Kraken/LeftRear/AppliedOutput",
    )
    right_front_card = MotorCardStatus(
        motor_id=MotorId.right_front,
        label="Right Front Drive",
        active=_wheel_is_active(telemetry.drive.right_front),
        enabled=telemetry.drive.right_front.enabled,
        current_a=telemetry.drive.right_front.torque_current_a,
        output_percent=telemetry.drive.right_front.applied_output,
        state=_wheel_state_text(telemetry.drive.right_front),
        source="Kraken/RightFront/AppliedOutput",
    )
    right_rear_card = MotorCardStatus(
        motor_id=MotorId.right_rear,
        label="Right Rear Drive",
        active=_wheel_is_active(telemetry.drive.right_rear),
        enabled=telemetry.drive.right_rear.enabled,
        current_a=telemetry.drive.right_rear.torque_current_a,
        output_percent=telemetry.drive.right_rear.applied_output,
        state=_wheel_state_text(telemetry.drive.right_rear),
        source="Kraken/RightRear/AppliedOutput",
    )
    excavator_card = MotorCardStatus(
        motor_id=MotorId.excavator,
        label="Excavator",
        active=_excavation_is_active(telemetry),
        enabled=_excavation_is_active(telemetry),
        current_a=_max_optional_float(
            telemetry.excavation.belt_leader_torque_current_a,
            telemetry.excavation.belt_follower_torque_current_a,
        ),
        output_percent=telemetry.excavation.belt_output,
        state=telemetry.excavation.state,
        source="Excav/BeltRunning",
    )
    deposition_card = MotorCardStatus(
        motor_id=MotorId.deposition,
        label="Deposition",
        active=_deposition_is_active(telemetry),
        enabled=_deposition_is_active(telemetry),
        current_a=_max_optional_float(
            telemetry.deposition.torque_current_a,
            telemetry.deposition.tailgate_torque_current_a,
        ),
        output_percent=telemetry.deposition.output,
        state=_compose_deposition_state(telemetry),
        source="Deposit/Running",
    )
    return MotorCardsTelemetry(
        drive=drive_card,
        left_front=left_front_card,
        left_rear=left_rear_card,
        right_front=right_front_card,
        right_rear=right_rear_card,
        excavator=excavator_card,
        deposition=deposition_card,
        conveyor=MotorCardStatus(
            motor_id=MotorId.deposition,
            label="Conveyor",
            active=deposition_card.active,
            enabled=deposition_card.enabled,
            current_a=deposition_card.current_a,
            output_percent=deposition_card.output_percent,
            state=deposition_card.state,
            source=deposition_card.source,
        ),
    )


def _drive_motor(values: Dict[str, Any], prefix: str) -> DriveMotorTelemetry:
    return DriveMotorTelemetry(
        velocity_tps=_optional_float(values.get(f"{prefix}/VelocityTps")),
        torque_current_a=_optional_float(values.get(f"{prefix}/TorqueCurrentA")),
        enabled=_optional_bool(values.get(f"{prefix}/Enabled")),
        applied_output=_optional_float(values.get(f"{prefix}/AppliedOutput")),
    )


def _has_any(values: Dict[str, Any], *keys: str) -> bool:
    return any(values.get(key) is not None for key in keys)


def _active_from_output(value: Optional[float], threshold: float = 1e-4) -> Optional[bool]:
    if value is None:
        return None
    return abs(value) > threshold


def _active_from_magnitude(value: Optional[float], threshold: float) -> Optional[bool]:
    if value is None:
        return None
    return abs(value) > threshold


def _any_known_true(*values: Optional[bool]) -> Optional[bool]:
    known = [value for value in values if value is not None]
    if not known:
        return None
    return any(known)


def _wheel_is_active(wheel: DriveMotorTelemetry) -> Optional[bool]:
    return _any_known_true(
        _active_from_output(wheel.applied_output),
        _active_from_magnitude(wheel.velocity_tps, 1e-3),
        _active_from_magnitude(wheel.torque_current_a, 0.5),
    )


def _excavation_is_active(telemetry: JetsonMotorTelemetry) -> Optional[bool]:
    if telemetry.excavation.belt_running is not None:
        return telemetry.excavation.belt_running
    return _any_known_true(
        _active_from_output(telemetry.excavation.belt_output),
        _active_from_magnitude(telemetry.excavation.belt_leader_torque_current_a, 0.5),
        _active_from_magnitude(telemetry.excavation.belt_follower_torque_current_a, 0.5),
    )


def _deposition_is_active(telemetry: JetsonMotorTelemetry) -> Optional[bool]:
    if telemetry.deposition.running is not None:
        return telemetry.deposition.running
    return _any_known_true(
        _active_from_output(telemetry.deposition.output),
        telemetry.deposition.collecting_assist,
        telemetry.deposition.depositing,
        telemetry.deposition.tailgate_moving,
        _active_from_magnitude(telemetry.deposition.torque_current_a, 0.5),
        _active_from_magnitude(telemetry.deposition.tailgate_torque_current_a, 0.5),
    )


def _max_optional_float(*values: Optional[float]) -> Optional[float]:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return max(numbers)


def _compose_deposition_state(telemetry: JetsonMotorTelemetry) -> Optional[str]:
    states: List[str] = []
    if telemetry.deposition.door_state:
        states.append(telemetry.deposition.door_state)
    if telemetry.deposition.tailgate_state:
        states.append(f"TAILGATE_{telemetry.deposition.tailgate_state}")
    elif telemetry.deposition.tailgate_open:
        states.append("TAILGATE_OPEN")
    elif telemetry.deposition.tailgate_closed:
        states.append("TAILGATE_CLOSED")
    if telemetry.deposition.collecting_assist:
        states.append("COLLECTING_ASSIST")
    if telemetry.deposition.depositing:
        states.append("DEPOSITING")
    if not states:
        return None
    return " | ".join(states)


def _drive_is_active(telemetry: JetsonMotorTelemetry) -> Optional[bool]:
    if telemetry.drive.active is not None:
        return telemetry.drive.active
    wheel_activity = [
        _wheel_is_active(telemetry.drive.left_front),
        _wheel_is_active(telemetry.drive.left_rear),
        _wheel_is_active(telemetry.drive.right_front),
        _wheel_is_active(telemetry.drive.right_rear),
    ]
    known = [value for value in wheel_activity if value is not None]
    if known:
        return any(known)
    side_activity = [
        _active_from_output(telemetry.drive.left_output),
        _active_from_output(telemetry.drive.right_output),
    ]
    known_side = [value for value in side_activity if value is not None]
    if known_side:
        return any(known_side)
    return None


def _drive_is_enabled(telemetry: JetsonMotorTelemetry) -> Optional[bool]:
    wheel_enabled = [
        telemetry.drive.left_front.enabled,
        telemetry.drive.left_rear.enabled,
        telemetry.drive.right_front.enabled,
        telemetry.drive.right_rear.enabled,
    ]
    known = [value for value in wheel_enabled if value is not None]
    if known:
        return any(known)
    return _drive_is_active(telemetry)


def _sum_optional_floats(*values: Optional[float]) -> Optional[float]:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers)


def _max_abs_optional_float(*values: Optional[float]) -> Optional[float]:
    numbers = [abs(value) for value in values if value is not None]
    if not numbers:
        return None
    return max(numbers)


def _drive_state_text(telemetry: JetsonMotorTelemetry) -> Optional[str]:
    active = _drive_is_active(telemetry)
    left = telemetry.drive.left_output
    right = telemetry.drive.right_output
    parts: List[str] = []
    if active is True:
        parts.append("ACTIVE")
    elif active is False:
        parts.append("IDLE")
    if left is not None:
        parts.append(f"L={left:.3f}")
    if right is not None:
        parts.append(f"R={right:.3f}")
    if telemetry.drive.wheel_test_mode:
        parts.append(f"TEST={telemetry.drive.wheel_test_mode}")
    if not parts:
        return None
    return " | ".join(parts)


def _wheel_state_text(wheel: DriveMotorTelemetry) -> Optional[str]:
    parts: List[str] = []
    if wheel.velocity_tps is not None:
        parts.append(f"{wheel.velocity_tps:.3f} tps")
    if wheel.applied_output is not None:
        parts.append(f"out={wheel.applied_output:.3f}")
    if not parts:
        return None
    return " | ".join(parts)


motor_service = MotorService()
