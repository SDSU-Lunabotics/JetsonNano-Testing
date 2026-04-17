from typing import List

from app.schemas.common import Fault, FaultCode, MotorId
from app.schemas.motors import (
    MotorHealth,
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
        motors: List[MotorHealth] = []

        for raw_motor in payload.get("motors", []):
            motor_id_raw = raw_motor.get("motor_id")
            if motor_id_raw is None:
                continue

            try:
                motor_id = MotorId(motor_id_raw)
            except ValueError:
                continue

            faults: List[Fault] = []
            if raw_motor.get("fault"):
                bridge_fault_code = raw_motor.get("fault_code")
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
                    enabled=bool(raw_motor.get("enabled")) if raw_motor.get("enabled") is not None else None,
                    current_a=_optional_float(raw_motor.get("current_a")),
                    rpm=_optional_float(raw_motor.get("rpm")),
                    torque_nm=None,
                    last_update_ms=raw_motor.get("last_update_ms") or timestamp,
                    faults=faults,
                )
            )

        return MotorsStatusResponse(
            timestamp_ms=timestamp,
            motors=motors,
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


def _optional_float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_int(value):
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


motor_service = MotorService()
