# just a stub file for now to match UI shape, but can be made real later

from __future__ import annotations

from app.schemas.wireless import (
    WirelessStatusResponse,
    DeviceWirelessStatus,
    WirelessRadioState,
)


class WirelessService:
    def __init__(self) -> None:
        self._status = WirelessStatusResponse(
            devices=[
                DeviceWirelessStatus(
                    device_name="Jetson Nano",
                    radio_state=WirelessRadioState(
                        band_24ghz=True,
                        band_5ghz=True,
                    ),
                    current_channel=1,
                    status_ok=True,
                    status_message="Wireless status placeholder",
                    comm_line="Wi-Fi",
                    activity="idle",
                    packet_metadata=[],
                    bandwidth=0.0,
                ),
                DeviceWirelessStatus(
                    device_name="RoboRIO",
                    radio_state=WirelessRadioState(
                        band_24ghz=False,
                        band_5ghz=False,
                    ),
                    current_channel=1,
                    status_ok=True,
                    status_message="Connected via control network placeholder",
                    comm_line="Ethernet",
                    activity="idle",
                    packet_metadata=[],
                    bandwidth=0.0,
                ),
            ],
            average_bandwidth=0.0,
            team_ssid="TEAM_00",
        )

    def get_status(self) -> WirelessStatusResponse:
        # keep timestamp fresh
        return WirelessStatusResponse(
            devices=self._status.devices,
            average_bandwidth=self._status.average_bandwidth,
            team_ssid=self._status.team_ssid,
        )


wireless_service = WirelessService()