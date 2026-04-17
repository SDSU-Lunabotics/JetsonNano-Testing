from __future__ import annotations
from typing import Dict, List, Literal, Union
from pydantic import BaseModel, Field
import time

WiFiChannel = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
ActivityState = Literal["idle", "transmitting", "receiving", "error"]

PacketMetadata = Dict[str, Union[str, int, float]]


class WirelessRadioState(BaseModel):
    band_24ghz: bool
    band_5ghz: bool


class DeviceWirelessStatus(BaseModel):
    device_name: str
    radio_state: WirelessRadioState
    available_channels: List[WiFiChannel] = Field(default_factory=lambda: list(range(1, 12)))
    current_channel: WiFiChannel
    status_ok: bool
    status_message: str = ""
    comm_line: str = "N/A"
    activity: ActivityState = "idle"
    packet_metadata: List[PacketMetadata] = Field(default_factory=list)
    bandwidth: float = 0.0


class WirelessStatusResponse(BaseModel):
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    devices: List[DeviceWirelessStatus] = Field(default_factory=list)
    average_bandwidth: float = 0.0
    team_ssid: str = "TEAM_00"


class WirelessConfigUpdateRequest(BaseModel):
    team_ssid: str = Field(min_length=1, max_length=128)


class WirelessConfigUpdateResponse(BaseModel):
    ok: bool = True
    team_ssid: str
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
