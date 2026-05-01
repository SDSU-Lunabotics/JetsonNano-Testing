import json
import os
import shutil
import subprocess
import time
from typing import Optional, List

from app.core.settings import settings
from app.schemas.common import Fault, FaultCode, LinkStats
from app.schemas.network import NetworkVerifyRequest, NetworkVerifyResponse, NetworkStatusResponse
from app.services.state_service import now_ms


class NetworkService:
    def __init__(self) -> None:
        self._link = LinkStats(
            latency_ms=None,
            packet_loss_pct=None,
            throughput_mbps=None,
        )
        self._last_verify_ms: Optional[int] = None

    def _target_host(self, target: Optional[str]) -> str:
        target = target or "rover"

        if target == "jetson":
            return settings.jetson_ip
        if target == "roborio":
            return settings.roborio_ip
        if target == "rover":
            return settings.roborio_ip
        if target == "camera":
            return settings.camera_host

        raise ValueError(f"Unknown target '{target}'")

    def _supports_throughput(self, target: str) -> bool:
        if not settings.throughput_test_enabled:
            return False

        return target in {"jetson", "camera", "roborio", "rover"}

    def _route_interface(self, host: str) -> Optional[str]:
        ip_path = shutil.which("ip")
        if not ip_path:
            return None

        try:
            result = subprocess.run(
                [ip_path, "route", "get", host],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split()
        for index, token in enumerate(parts):
            if token == "dev" and index + 1 < len(parts):
                return parts[index + 1]

        return None

    def _read_interface_bytes(self, interface: str) -> Optional[int]:
        stats_path = f"/sys/class/net/{interface}/statistics"
        rx_path = os.path.join(stats_path, "rx_bytes")
        tx_path = os.path.join(stats_path, "tx_bytes")

        try:
            with open(rx_path, "r", encoding="utf-8") as fh:
                rx_bytes = int(fh.read().strip())
            with open(tx_path, "r", encoding="utf-8") as fh:
                tx_bytes = int(fh.read().strip())
        except Exception:
            return None

        return rx_bytes + tx_bytes

    def _measure_roborio_estimated_throughput(self, host: str) -> Optional[float]:
        interface = self._route_interface(host)
        if not interface:
            return None

        before_bytes = self._read_interface_bytes(interface)
        if before_bytes is None:
            return None

        ping_count = max(1, int(settings.roborio_estimate_ping_count))
        timeout_ms = max(100, int(settings.roborio_estimate_timeout_ms))
        timeout_s = max(timeout_ms / 1000.0, 0.1)

        start = time.perf_counter()
        try:
            subprocess.run(
                ["ping", "-c", str(ping_count), "-W", str(max(1, int(timeout_s))), host],
                capture_output=True,
                text=True,
                timeout=(timeout_s * ping_count) + 1.0,
            )
        except Exception:
            return None

        elapsed_s = time.perf_counter() - start
        if elapsed_s <= 0.0:
            return None

        after_bytes = self._read_interface_bytes(interface)
        if after_bytes is None or after_bytes < before_bytes:
            return None

        bits_per_second = ((after_bytes - before_bytes) * 8.0) / elapsed_s
        return round(bits_per_second / 1_000_000.0, 2)

    def _measure_throughput(self, host: str, target: str, timeout_ms: int) -> Optional[float]:
        if not self._supports_throughput(target):
            return None

        if target in {"roborio", "rover"}:
            return self._measure_roborio_estimated_throughput(host)

        iperf3_path = shutil.which("iperf3")
        if not iperf3_path:
            return None

        duration_s = max(1, int(settings.throughput_test_duration_s))
        timeout_s = max(timeout_ms / 1000.0, 0.1)

        try:
            result = subprocess.run(
                [
                    iperf3_path,
                    "-c",
                    host,
                    "-p",
                    str(settings.throughput_test_port),
                    "-J",
                    "-t",
                    str(duration_s),
                    "--connect-timeout",
                    str(timeout_ms),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s + duration_s + 2.0,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

        end = payload.get("end", {})
        bits_per_second = None

        for path in (("sum_received",), ("sum_sent",), ("sum",)):
            node = end
            for part in path:
                node = node.get(part, {}) if isinstance(node, dict) else {}
            if isinstance(node, dict) and node.get("bits_per_second") is not None:
                bits_per_second = node.get("bits_per_second")
                break

        if bits_per_second is None:
            return None

        try:
            return round(float(bits_per_second) / 1_000_000.0, 2)
        except (TypeError, ValueError):
            return None

    def get_link_stats(self) -> LinkStats:
        self._refresh_link_stats_if_stale()
        return self._link

    def _refresh_link_stats_if_stale(self, target: str = "rover") -> None:
        now = now_ms()
        if self._last_verify_ms is not None and (now - self._last_verify_ms) < settings.network_status_ttl_ms:
            return

        self._run_verify(target=target, timeout_ms=2000)

    def _current_faults(self) -> Optional[List[Fault]]:
        faults: List[Fault] = []
        now = now_ms()

        if self._link.latency_ms is not None and self._link.latency_ms > settings.max_latency_ms:
            faults.append(
                Fault(
                    code=FaultCode.NETWORK_HIGH_LATENCY,
                    severity="warn",
                    message=f"High latency detected: {self._link.latency_ms} ms",
                    source="network",
                    timestamp_ms=now,
                )
            )

        if self._link.packet_loss_pct is not None and self._link.packet_loss_pct > settings.max_packet_loss_pct:
            faults.append(
                Fault(
                    code=FaultCode.NETWORK_HIGH_PACKET_LOSS,
                    severity="warn",
                    message=f"High packet loss detected: {self._link.packet_loss_pct}%",
                    source="network",
                    timestamp_ms=now,
                )
            )

        return faults or None

    def get_network_status(self) -> NetworkStatusResponse:
        self._refresh_link_stats_if_stale()
        ok = True
        if self._link.latency_ms is not None and self._link.latency_ms > settings.max_latency_ms:
            ok = False
        if self._link.packet_loss_pct is not None and self._link.packet_loss_pct > settings.max_packet_loss_pct:
            ok = False

        return NetworkStatusResponse(
            timestamp_ms=now_ms(),
            link=self._link,
            ok=ok,
            faults=self._current_faults(),
        )

    def verify(self, req: NetworkVerifyRequest) -> NetworkVerifyResponse:
        return self._run_verify(target=req.target or "rover", timeout_ms=req.timeout_ms or 5000)

    def _run_verify(self, target: str, timeout_ms: int) -> NetworkVerifyResponse:
        host = self._target_host(target)
        timeout_s = max(timeout_ms / 1000.0, 0.1)

        try:
            start = time.perf_counter()
            result = subprocess.run(
                ["ping", "-c", "1", "-W", str(max(1, int(timeout_s))), host],
                capture_output=True,
                text=True,
                timeout=timeout_s + 1.0,
            )
            latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
            self._last_verify_ms = now_ms()

            if result.returncode == 0:
                self._link.latency_ms = latency_ms
                self._link.packet_loss_pct = 0.0
                self._link.throughput_mbps = self._measure_throughput(
                    host=host,
                    target=target,
                    timeout_ms=timeout_ms,
                )

                return NetworkVerifyResponse(
                    ok=True,
                    timestamp_ms=now_ms(),
                    link=self._link,
                    message=f"Verified {target} link",
                )

            self._link.latency_ms = None
            self._link.packet_loss_pct = 100.0
            self._link.throughput_mbps = None

            return NetworkVerifyResponse(
                ok=False,
                timestamp_ms=now_ms(),
                link=self._link,
                message=f"Failed to reach {target}",
            )

        except Exception as e:
            self._last_verify_ms = now_ms()
            self._link.latency_ms = None
            self._link.packet_loss_pct = None
            self._link.throughput_mbps = None

            return NetworkVerifyResponse(
                ok=False,
                timestamp_ms=now_ms(),
                link=self._link,
                message=str(e),
            )


network_service = NetworkService()
