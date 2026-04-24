from app.schemas.common import BatteryStatus
from app.services.roborio_bridge_service import roborio_bridge_service


class BatteryService:
    def get_battery_status(self) -> BatteryStatus:
        source_key = "Battery/Voltage"
        bridge_status = roborio_bridge_service.get_status()
        bridge_connected = bool(bridge_status.get("connected", False))
        raw_value = (bridge_status.get("values") or {}).get(source_key)
        voltage = raw_value

        if voltage is None:
            return BatteryStatus(
                voltage_v=None,
                low=None,
                source_key=source_key,
                raw_value=raw_value,
                bridge_connected=bridge_connected,
            )

        try:
            voltage = float(voltage)
        except Exception:
            voltage = None

        return BatteryStatus(
            voltage_v=voltage,
            low=(voltage < 11.5 if voltage is not None else None),
            source_key=source_key,
            raw_value=raw_value,
            bridge_connected=bridge_connected,
        )


battery_service = BatteryService()
