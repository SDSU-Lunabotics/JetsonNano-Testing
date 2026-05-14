from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.settings import settings
from app.schemas.wireless import (
    DeviceWirelessStatus,
    WirelessConfigUpdateResponse,
    WirelessRadioState,
    WirelessStatusResponse,
)
from app.services.roborio_bridge_service import roborio_bridge_service
from app.services.state_service import now_ms


class WirelessService:
    def __init__(self) -> None:
        self._state_path = Path(settings.wireless_state_file)
        self._team_ssid = "TEAM_24"
        self._bandwidth_samples: Dict[str, Tuple[int, int]] = {}
        self._load_state()

    def get_status(self) -> WirelessStatusResponse:
        jetson_status = self._build_jetson_status()
        roborio_status = self._build_roborio_status()
        average_bandwidth = (jetson_status.bandwidth + roborio_status.bandwidth) / 2.0

        return WirelessStatusResponse(
            devices=[jetson_status, roborio_status],
            average_bandwidth=average_bandwidth,
            team_ssid=self._team_ssid,
        )

    def update_team_ssid(self, team_ssid: str) -> WirelessConfigUpdateResponse:
        normalized = team_ssid.strip()
        if not normalized:
            raise ValueError("team_ssid cannot be empty")

        self._team_ssid = normalized
        self._save_state()
        return WirelessConfigUpdateResponse(team_ssid=self._team_ssid)

    def _build_jetson_status(self) -> DeviceWirelessStatus:
        iface, ssid, channel = self._read_wifi_info()
        bandwidth = self._measure_bandwidth(iface)
        connected = bool(iface and ssid)

        band_24ghz = channel is not None and 1 <= channel <= 14
        band_5ghz = channel is not None and channel > 14

        packet_metadata: List[Dict[str, object]] = []
        if iface:
            packet_metadata.append({"interface": iface})
        if ssid:
            packet_metadata.append({"ssid": ssid})
        if channel is not None:
            packet_metadata.append({"channel": channel})

        status_message = f"Connected to {ssid}" if connected else "Wi-Fi details unavailable"
        comm_line = "Wi-Fi" if iface else "Unknown"
        activity = "transmitting" if bandwidth > 0.0 else "idle"

        return DeviceWirelessStatus(
            device_name="Jetson Nano",
            radio_state=WirelessRadioState(
                band_24ghz=band_24ghz,
                band_5ghz=band_5ghz,
            ),
            current_channel=_coerce_channel(channel),
            status_ok=connected,
            status_message=status_message,
            comm_line=comm_line,
            activity=activity,
            packet_metadata=packet_metadata,
            bandwidth=bandwidth,
        )

    def _build_roborio_status(self) -> DeviceWirelessStatus:
        connected = roborio_bridge_service.is_connected()
        status_message = "Connected via control network" if connected else "RoboRIO bridge not connected"

        return DeviceWirelessStatus(
            device_name="RoboRIO",
            radio_state=WirelessRadioState(
                band_24ghz=False,
                band_5ghz=False,
            ),
            current_channel=1,
            status_ok=connected,
            status_message=status_message,
            comm_line="Ethernet",
            activity="idle",
            packet_metadata=[],
            bandwidth=0.0,
        )

    def _load_state(self) -> None:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return

        team_ssid = str(payload.get("team_ssid", "")).strip()
        if team_ssid:
            self._team_ssid = team_ssid

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "team_ssid": self._team_ssid,
            "updated_ms": now_ms(),
        }
        self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read_wifi_info(self) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        ssid = self._run_command(["iwgetid", "-r"])
        channel = None

        iface_name = self._wifi_interface_from_iw()

        channel_text = self._run_command(["iwgetid", "-c"])
        if channel_text:
            match = re.search(r"Channel:(\d+)", channel_text)
            if match:
                channel = int(match.group(1))

        if not ssid or channel is None:
            nmcli = self._read_nmcli_wifi()
            iface_name = iface_name or nmcli.get("iface")
            ssid = ssid or nmcli.get("ssid")
            channel = channel if channel is not None else nmcli.get("channel")

        return iface_name, ssid, channel

    def _read_nmcli_wifi(self) -> Dict[str, object]:
        output = self._run_command(["nmcli", "-t", "-f", "DEVICE,ACTIVE,SSID,CHAN", "dev", "wifi"])
        if not output:
            return {}

        for line in output.splitlines():
            parts = line.split(":")
            if len(parts) < 4 or parts[1] != "yes":
                continue
            channel = None
            try:
                channel = int(parts[3])
            except ValueError:
                channel = None
            return {
                "iface": parts[0] or None,
                "ssid": parts[2] or None,
                "channel": channel,
            }
        return {}

    def _wifi_interface_from_iw(self) -> Optional[str]:
        output = self._run_command(["iw", "dev"])
        if not output:
            return None

        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("Interface "):
                return stripped.split(" ", 1)[1].strip() or None
        return None

    def _measure_bandwidth(self, iface: Optional[str]) -> float:
        if not iface:
            return 0.0

        base = Path("/sys/class/net") / iface / "statistics"
        try:
            rx = int((base / "rx_bytes").read_text(encoding="utf-8").strip())
            tx = int((base / "tx_bytes").read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError, OSError):
            return 0.0

        now = now_ms()
        previous = self._bandwidth_samples.get(iface)
        self._bandwidth_samples[iface] = (now, rx + tx)
        if previous is None:
            return 0.0

        previous_ms, previous_total = previous
        delta_ms = max(now - previous_ms, 1)
        delta_bytes = max((rx + tx) - previous_total, 0)
        bits_per_second = (delta_bytes * 8.0 * 1000.0) / float(delta_ms)
        return bits_per_second / 1_000_000.0

    def _run_command(self, args: List[str]) -> Optional[str]:
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None

        if completed.returncode != 0:
            return None

        output = completed.stdout.strip()
        return output or None


def _coerce_channel(channel: Optional[int]) -> int:
    if channel is None:
        return 1
    return max(1, min(11, int(channel)))


wireless_service = WirelessService()
