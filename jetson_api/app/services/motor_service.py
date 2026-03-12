from typing import Dict, List

from app.schemas.common import MotorId
from app.schemas.motors import (
    MotorHealth,
    MotorsStatusResponse,
    MotorCommandRequest,
    MotorCommandResponse,
)
from app.services.state_service import now_ms


class MotorService:
    def __init__(self) -> None:
        self._motors: Dict[str, MotorHealth] = {
            "left_front": MotorHealth(motor_id=MotorId.left_front, enabled=True, faults=[]),
            "right_front": MotorHealth(motor_id=MotorId.right_front, enabled=True, faults=[]),
            "left_rear": MotorHealth(motor_id=MotorId.left_rear, enabled=True, faults=[]),
            "right_rear": MotorHealth(motor_id=MotorId.right_rear, enabled=True, faults=[]),
            "excavator": MotorHealth(motor_id=MotorId.excavator, enabled=True, faults=[]),
            "deposition": MotorHealth(motor_id=MotorId.deposition, enabled=True, faults=[]),
        }

    def get_status(self) -> MotorsStatusResponse:
        timestamp = now_ms()
        motors: List[MotorHealth] = []

        for motor in self._motors.values():
            motor.last_update_ms = timestamp
            motors.append(motor)

        return MotorsStatusResponse(
            timestamp_ms=timestamp,
            motors=motors,
        )

    def command_motor(self, motor_id: MotorId, req: MotorCommandRequest) -> MotorCommandResponse:
        if motor_id.value not in self._motors:
            raise ValueError(f"Unknown motor_id '{motor_id.value}'")

        motor = self._motors[motor_id.value]

        if req.mode == "stop":
            motor.rpm = 0.0
            motor.torque_nm = None
        elif req.mode == "rpm":
            if req.value is None:
                raise ValueError("value required for rpm mode")
            motor.rpm = float(req.value)
        elif req.mode == "percent":
            if req.value is None:
                raise ValueError("value required for percent mode")
            motor.rpm = None
        else:
            raise ValueError("Invalid motor command mode")

        motor.last_update_ms = now_ms()

        return MotorCommandResponse(
            ok=True,
            motor_id=motor_id.value,
            timestamp_ms=now_ms(),
        )


motor_service = MotorService()