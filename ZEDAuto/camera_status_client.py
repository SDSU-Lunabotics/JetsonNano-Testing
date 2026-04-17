from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request


class CameraStatusHeartbeat:
    def __init__(
        self,
        heartbeat_url: str,
        *,
        backend: str = "zed",
        source: str = "zed_ground_wall",
        interval_ms: int = 1000,
        timeout_ms: int = 250,
        streaming: bool = False,
    ) -> None:
        self.heartbeat_url = heartbeat_url
        self.backend = backend
        self.source = source
        self.interval_ms = max(int(interval_ms), 250)
        self.timeout_s = max(int(timeout_ms), 1) / 1000.0
        self.streaming = bool(streaming)

        self._stop_event = threading.Event()
        self._last_error_log_ms = 0
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="camera-status-heartbeat",
            daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=1.0)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._send_heartbeat()
            self._stop_event.wait(self.interval_ms / 1000.0)

    def _send_heartbeat(self) -> None:
        payload = json.dumps(
            {
                "backend": self.backend,
                "source": self.source,
                "streaming": self.streaming,
                "timestamp_ms": int(time.time() * 1000),
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.heartbeat_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                response.read(1)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._rate_limited_log(f"Camera heartbeat failed: {exc}")

    def _rate_limited_log(self, message: str) -> None:
        now_ms = int(time.time() * 1000)
        if (now_ms - self._last_error_log_ms) >= 5000:
            print(message)
            self._last_error_log_ms = now_ms
