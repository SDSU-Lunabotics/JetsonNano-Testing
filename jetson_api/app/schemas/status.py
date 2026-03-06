from pydantic import BaseModel


class JetsonStatusResponse(BaseModel):
    ok: bool
    led_state: bool
    led_override: bool
    automation_enabled: bool
    estop: bool
    excavator_running: bool
    conveyor_running: bool
    timestamp_ms: int