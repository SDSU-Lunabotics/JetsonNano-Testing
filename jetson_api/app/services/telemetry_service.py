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


class TelemetryService:
    def __init__(self) -> None:
        self._jetson_last_seen_ms: Optional[int] = now_ms()
        self._roborio_last_seen_ms: Optional[int] = None
        self._lidar_last_seen_ms: Optional[int] = None

    def update_jetson_heartbeat(self) -> None:
        self._jetson_last_seen_ms = now_ms()

    def update_roborio_heartbeat(self) -> None:
        self._roborio_last_seen_ms = now_ms()

    def update_lidar_heartbeat(self) -> None:
        self._lidar_last_seen_ms = now_ms()

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

    def get_status(self) -> StatusResponse:
        now = now_ms()
        self._jetson_last_seen_ms = now

        bridge_status = roborio_bridge_service.get_status()

        if bool(bridge_status.get("connected", False)):
            self._roborio_last_seen_ms = now

        rover = RoverStatus(
            heartbeat=self._hb(self._jetson_last_seen_ms, now),
            jetson=self._hb(self._jetson_last_seen_ms, now),
            roborio=self._hb(self._roborio_last_seen_ms, now),
            lidar=self._hb(self._lidar_last_seen_ms, now),
        )

        control = ControlStatus(
            armed=not state_service.estop,
            estop=state_service.estop,
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
            camera=camera_service.get_status(),
            wireless=wireless_service.get_status(),
        )


telemetry_service = TelemetryService()
