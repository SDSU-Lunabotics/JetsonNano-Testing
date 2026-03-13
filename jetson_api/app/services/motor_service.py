from typing import List

from app.schemas.common import MotorId
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
        timestamp = now_ms()

        values = roborio_bridge_service.get_status().get("values", {})

        def num(key: str):
            value = values.get(key)
            try:
                return float(value) if value is not None else None
            except Exception:
                return None

        def boolean(key: str):
            value = values.get(key)
            if isinstance(value, bool):
                return value
            return None

        motors: List[MotorHealth] = [
            MotorHealth(
                motor_id=MotorId.left_front,
                enabled=boolean("Kraken/LeftFront/Enabled"),
                current_a=num("Kraken/LeftFront/TorqueCurrentA"),
                rpm=None,
                torque_nm=None,
                last_update_ms=timestamp,
                faults=[],
            ),
            MotorHealth(
                motor_id=MotorId.left_rear,
                enabled=boolean("Kraken/LeftRear/Enabled"),
                current_a=num("Kraken/LeftRear/TorqueCurrentA"),
                rpm=None,
                torque_nm=None,
                last_update_ms=timestamp,
                faults=[],
            ),
            MotorHealth(
                motor_id=MotorId.right_front,
                enabled=boolean("Kraken/RightFront/Enabled"),
                current_a=num("Kraken/RightFront/TorqueCurrentA"),
                rpm=None,
                torque_nm=None,
                last_update_ms=timestamp,
                faults=[],
            ),
            MotorHealth(
                motor_id=MotorId.right_rear,
                enabled=boolean("Kraken/RightRear/Enabled"),
                current_a=num("Kraken/RightRear/TorqueCurrentA"),
                rpm=None,
                torque_nm=None,
                last_update_ms=timestamp,
                faults=[],
            ),
            MotorHealth(
                motor_id=MotorId.excavator,
                enabled=None,
                current_a=None,
                rpm=None,
                torque_nm=None,
                last_update_ms=timestamp,
                faults=[],
            ),
            MotorHealth(
                motor_id=MotorId.deposition,
                enabled=None,
                current_a=None,
                rpm=None,
                torque_nm=None,
                last_update_ms=timestamp,
                faults=[],
            ),
        ]

        return MotorsStatusResponse(
            timestamp_ms=timestamp,
            motors=motors,
        )

    def command_motor(self, motor_id: MotorId, req: MotorCommandRequest) -> MotorCommandResponse:
        # Placeholder until direct motor command wiring is added
        return MotorCommandResponse(
            ok=True,
            motor_id=motor_id.value,
            timestamp_ms=now_ms(),
        )


motor_service = MotorService()