from typing import Optional

from app.schemas.common import Heartbeat, ControllerStatus
from app.schemas.health import StatusResponse
from app.schemas.status import RoverStatus, ControlStatus
from app.services.state_service import state_service, now_ms
from app.services.network_service import network_service
from app.services.battery_service import battery_service
from app.services.wireless_service import wireless_service


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

    def get_status(self) -> StatusResponse:
        now = now_ms()

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
            controller=ControllerStatus(
                connected=False,
                last_input_ms=None,
                last_input=None,
            ),
        )

        return StatusResponse(
            timestamp_ms=now,
            rover=rover,
            link=network_service.get_link_stats(),
            battery=battery_service.get_battery_status(),
            control=control,
            wireless=wireless_service.get_status(),
        )


telemetry_service = TelemetryService()