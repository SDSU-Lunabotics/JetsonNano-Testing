import time


def now_ms() -> int:
    return int(time.time() * 1000)


class StateService:
    def __init__(self) -> None:
        self.led_state = False
        self.led_override = True
        self.automation_enabled = False
        self.estop = False
        self.excavator_running = False
        self.conveyor_running = False

    def status_dict(self) -> dict:
        return {
            "ok": True,
            "led_state": self.led_state,
            "led_override": self.led_override,
            "automation_enabled": self.automation_enabled,
            "estop": self.estop,
            "excavator_running": self.excavator_running,
            "conveyor_running": self.conveyor_running,
            "timestamp_ms": now_ms(),
        }


state_service = StateService()