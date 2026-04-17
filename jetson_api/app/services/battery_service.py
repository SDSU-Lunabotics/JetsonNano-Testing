from app.schemas.common import BatteryStatus
from app.services.roborio_bridge_service import roborio_bridge_service


class BatteryService:
    def get_battery_status(self) -> BatteryStatus:
        voltage = roborio_bridge_service.get_value("Battery/Voltage", None)

        if voltage is None:
            return BatteryStatus(
                voltage_v=None,
                low=None,
            )

        try:
            voltage = float(voltage)
        except Exception:
            voltage = None

        return BatteryStatus(
            voltage_v=voltage,
            low=(voltage < 11.5 if voltage is not None else None),
        )


battery_service = BatteryService()