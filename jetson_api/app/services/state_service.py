import time


def now_ms() -> int:
    return int(time.time() * 1000)


class StateService:
    def __init__(self) -> None:
        self.led_state = False
        self.led_override = True

        # True autonomy state (reserve this for future real autonomy mode)
        self.autonomy_enabled = False

        # Manual motion/debug state
        self.manual_motion_active = False
        self.last_drive_speed = None
        self.last_drive_duration = None
        self.last_drive_timestamp_ms = None

        self.estop = False
        self.excavator_running = False
        self.deposition_running = False

    def status_dict(self) -> dict:
        return {
            "ok": True,
            "led_state": self.led_state,
            "led_override": self.led_override,
            "autonomy_enabled": self.autonomy_enabled,
            "manual_motion_active": self.manual_motion_active,
            "last_drive_speed": self.last_drive_speed,
            "last_drive_duration": self.last_drive_duration,
            "last_drive_timestamp_ms": self.last_drive_timestamp_ms,
            "estop": self.estop,
            "excavator_running": self.excavator_running,
            "deposition_running": self.deposition_running,
            # Keep the legacy key for older clients while the UI finishes migrating.
            "conveyor_running": self.deposition_running,
            "timestamp_ms": now_ms(),
        }


state_service = StateService()
