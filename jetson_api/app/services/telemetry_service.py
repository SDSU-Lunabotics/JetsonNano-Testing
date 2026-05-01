from typing import Optional

from app.schemas.common import Heartbeat, ControllerInput, ControllerStatus
from app.schemas.health import StatusResponse
from app.schemas.status import RoverStatus, ControlStatus
from app.services.state_service import state_service, now_ms
from app.services.network_service import network_service
from app.services.battery_service import battery_service
from app.services.wireless_service import wireless_service
from app.services.roborio_bridge_service import roborio_bridge_service
from app.services.camera_service import camera_service
from app.services.motor_service import motor_service
from app.services.nt_service import nt_service


class TelemetryService:
    def __init__(self) -> None:
        self._jetson_last_seen_ms: Optional[int] = now_ms()
        self._roborio_last_seen_ms: Optional[int] = None

    def update_jetson_heartbeat(self) -> None:
        self._jetson_last_seen_ms = now_ms()

    def update_roborio_heartbeat(self) -> None:
        self._roborio_last_seen_ms = now_ms()

    def _hb(self, last_seen_ms: Optional[int], now: int) -> Heartbeat:
        connected = last_seen_ms is not None and (now - last_seen_ms) < 5000
        return Heartbeat(
            connected=connected,
            last_seen_ms=last_seen_ms,
            age_ms=(None if last_seen_ms is None else now - last_seen_ms),
        )

    def _controller_status(self, payload) -> ControllerStatus:
        raw = payload.get("controller") or {}

        raw_input = raw.get("last_input")
        last_input = None
        if raw_input:
            try:
                last_input = ControllerInput(str(raw_input))
            except ValueError:
                last_input = None

        last_input_ms = raw.get("last_input_ms")
        if last_input_ms is not None:
            try:
                last_input_ms = int(last_input_ms)
            except (TypeError, ValueError):
                last_input_ms = None

        return ControllerStatus(
            connected=bool(raw.get("connected", False)),
            last_input_ms=last_input_ms,
            last_input=last_input,
        )

    def _bool_value(self, value) -> Optional[bool]:
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

    def _armed_status(self, bridge_status) -> bool:
        if self._combined_estop(bridge_status):
            return False

        if not bool(bridge_status.get("connected", False)):
            return False

        values = bridge_status.get("values") or {}
        enabled = self._bool_value(values.get("RoboRIO/Enabled"))
        if enabled is not None:
            return enabled

        driver_station_attached = self._bool_value(values.get("RoboRIO/DriverStationAttached"))
        return bool(driver_station_attached)

    def _combined_estop(self, bridge_status) -> bool:
        if state_service.estop:
            return True

        values = bridge_status.get("values") or {}
        roborio_estop = self._bool_value(values.get("Jetson/EStop"))
        return bool(roborio_estop)

    def get_status(self) -> StatusResponse:
        now = now_ms()
        self._jetson_last_seen_ms = now

        bridge_status = roborio_bridge_service.get_status()
        bridge_connected = bool(bridge_status.get("connected", False))
        bridge_heartbeat = bridge_status.get("heartbeat") or {}
        bridge_heartbeat_fresh = bool(bridge_heartbeat.get("fresh", False))

        # Fallback: if the local RoboRIO HTTP bridge is unavailable, use direct
        # NT connectivity so status still reflects live RoboRIO link health.
        if not bridge_connected:
            try:
                bridge_connected = nt_service.is_connected()
            except Exception:
                bridge_connected = False
            if bridge_connected:
                bridge_status["connected"] = True

        # UI connectivity should reflect the live bridge/NT link, even if the
        # optional heartbeat key is stale or temporarily missing.
        if bridge_connected:
            self._roborio_last_seen_ms = now

        if bridge_heartbeat_fresh:
            try:
                self._roborio_last_seen_ms = int(bridge_heartbeat.get("last_seen_ms"))
            except (TypeError, ValueError):
                self._roborio_last_seen_ms = now
        elif bridge_connected and not bridge_heartbeat:
            # Older bridge versions did not expose heartbeat freshness. Preserve
            # the NetworkTables-based behavior for that case only.
            self._roborio_last_seen_ms = now

        rover = RoverStatus(
            heartbeat=self._hb(self._jetson_last_seen_ms, now),
            jetson=self._hb(self._jetson_last_seen_ms, now),
            roborio=self._hb(self._roborio_last_seen_ms, now),
        )

        control = ControlStatus(
            armed=self._armed_status(bridge_status),
            estop=self._combined_estop(bridge_status),
            mode="manual",
            autonomy_running=state_service.autonomy_enabled,
            controller=self._controller_status(bridge_status),
        )

        return StatusResponse(
            timestamp_ms=now,
            rover=rover,
            link=network_service.get_link_stats(),
            battery=battery_service.get_battery_status(),
            control=control,
            motors=motor_service.get_status(),
            camera=camera_service.get_cached_status(),
            wireless=wireless_service.get_status(),
        )


telemetry_service = TelemetryService()
