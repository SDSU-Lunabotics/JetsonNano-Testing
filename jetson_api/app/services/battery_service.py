from app.schemas.common import BatteryStatus
from app.services.state_service import now_ms


class BatteryService:
    def __init__(self) -> None:
        self._battery = BatteryStatus(
            voltage_v=None,
            low=None,
        )

    def set_battery(self, voltage_v: float | None) -> None:
        self._battery.voltage_v = voltage_v
        if voltage_v is None:
            self._battery.low = None
        else:
            self._battery.low = voltage_v < 11.5

    def get_battery_status(self) -> BatteryStatus:
        return self._battery


battery_service = BatteryService()