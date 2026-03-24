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
        if target == "lidar":
            return settings.lidar_host

        raise ValueError(f"Unknown target '{target}'")

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
                self._link.throughput_mbps = None

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
