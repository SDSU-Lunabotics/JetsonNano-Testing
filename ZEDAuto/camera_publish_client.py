from __future__ import annotations

import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2


class HttpCameraPublisher:
    def __init__(
        self,
        publish_url: str,
        *,
        interval_ms: int = 120,
        jpeg_quality: int = 75,
        timeout_ms: int = 250,
        source: str = "zed_ground_wall",
    ) -> None:
        self.publish_url = publish_url
        self.interval_ms = max(int(interval_ms), 50)
        self.jpeg_quality = max(30, min(int(jpeg_quality), 100))
        self.timeout_s = max(int(timeout_ms), 1) / 1000.0
        self.source = source

        self._frame_condition = threading.Condition()
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._latest_frame_seq = 0
        self._last_sent_seq = 0
        self._last_error_log_ms = 0
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="zed-camera-publisher",
            daemon=True,
        )
        self._worker.start()

    def push_frame(self, frame) -> None:
        if frame is None:
            return

        with self._frame_condition:
            self._latest_frame = frame.copy()
            self._latest_frame_seq += 1
            self._frame_condition.notify_all()

    def stop(self) -> None:
        self._stop_event.set()
        with self._frame_condition:
            self._frame_condition.notify_all()
        self._worker.join(timeout=1.0)

    def _worker_loop(self) -> None:
        next_send_at = 0.0

        while not self._stop_event.is_set():
            with self._frame_condition:
                while (
                    not self._stop_event.is_set()
                    and self._latest_frame_seq <= self._last_sent_seq
                ):
                    self._frame_condition.wait(timeout=0.5)

                if self._stop_event.is_set():
                    return

                now = time.monotonic()
                if now < next_send_at:
                    self._frame_condition.wait(timeout=next_send_at - now)
                    continue

                frame = self._latest_frame.copy()
                frame_seq = self._latest_frame_seq

            self._publish_frame(frame)
            self._last_sent_seq = frame_seq
            next_send_at = time.monotonic() + (self.interval_ms / 1000.0)

    def _publish_frame(self, frame) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            self._rate_limited_log("Failed to encode camera view as JPEG for HTTP publish.")
            return

        query = urllib.parse.urlencode(
            {
                "source": self.source,
                "timestamp_ms": int(time.time() * 1000),
            }
        )
        target = self._append_query_params(self.publish_url, query)
        req = urllib.request.Request(
            target,
            data=encoded.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                response.read(1)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._rate_limited_log(f"Camera publish failed: {exc}")

    def _rate_limited_log(self, message: str) -> None:
        now_ms = int(time.time() * 1000)
        if (now_ms - self._last_error_log_ms) >= 5000:
            print(message)
            self._last_error_log_ms = now_ms

    @staticmethod
    def _append_query_params(base_url: str, query: str) -> str:
        parts = urllib.parse.urlsplit(base_url)
        merged_query = query if not parts.query else f"{parts.query}&{query}"
        return urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, parts.path, merged_query, parts.fragment)
        )
