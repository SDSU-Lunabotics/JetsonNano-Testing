from __future__ import annotations

import time
import requests
from typing import Any, Dict
from app.core.settings import settings


class RoboRIOBridgeService:
    def __init__(self) -> None:
        self._base_url = settings.roborio_bridge_url
        self._status_cache: Dict[str, Any] | None = None
        self._status_cache_expires_at: float = 0.0

    def _get(self, path: str) -> Dict[str, Any]:
        r = requests.get(f"{self._base_url}{path}", timeout=2)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self._base_url}{path}", json=payload, timeout=2)
        r.raise_for_status()
        return r.json()

    def get_status(self, *, force_refresh: bool = False) -> Dict[str, Any]:
        now = time.monotonic()
        if not force_refresh and self._status_cache is not None and now < self._status_cache_expires_at:
            return dict(self._status_cache)

        try:
            payload = self._get("/status")
        except requests.RequestException as exc:
            payload = {
                "status": "error",
                "timestamp_ms": 0,
                "connected": False,
                "values": {},
                "warnings": [],
                "error": str(exc),
                "bridge_url": self._base_url,
            }

        self._status_cache = dict(payload)
        self._status_cache_expires_at = time.monotonic() + 0.2
        return dict(payload)

    def is_connected(self) -> bool:
        payload = self.get_status()
        return bool(payload.get("connected", False))

    def get_value(self, key: str, default: Any = None) -> Any:
        payload = self.get_status()
        return payload.get("values", {}).get(key, default)

    def get_motors_status(self) -> Dict[str, Any]:
        status_payload = self.get_status()
        values = status_payload.get("values") or {}
        warnings = status_payload.get("warnings") or []

        if values:
            return {
                "timestamp_ms": status_payload.get("timestamp_ms", 0),
                "motors": [],
                "values": values,
                "warnings": warnings,
            }

        try:
            return self._get("/motors/status")
        except requests.RequestException:
            return {
                "timestamp_ms": 0,
                "motors": [],
                "values": {},
                "warnings": [],
            }

    def set_value(self, key: str, value: Any) -> Dict[str, Any]:
        try:
            return self._post("/set", {"key": key, "value": value})
        except requests.HTTPError as exc:
            message = None
            response = exc.response
            if response is not None:
                try:
                    message = response.json().get("message")
                except ValueError:
                    message = response.text or None
            raise ValueError(message or f"Failed to set '{key}'") from exc
        except requests.RequestException as exc:
            raise ValueError("RoboRIO bridge unavailable") from exc

    def set_estop(self, engage: bool) -> Dict[str, Any]:
        return self.set_value("Jetson/EStop", bool(engage))

    def command_motor(
        self,
        motor_id: str,
        mode: str,
        value: Any = None,
        duration_ms: Any = None,
    ) -> Dict[str, Any]:
        payload = {
            "mode": mode,
            "value": value,
            "duration_ms": duration_ms,
        }

        try:
            return self._post(f"/motors/{motor_id}/command", payload)
        except requests.HTTPError as exc:
            message = None
            response = exc.response
            if response is not None:
                try:
                    message = response.json().get("message")
                except ValueError:
                    message = response.text or None
            raise ValueError(message or f"Motor command failed for '{motor_id}'") from exc
        except requests.RequestException as exc:
            raise ValueError("RoboRIO bridge unavailable") from exc


roborio_bridge_service = RoboRIOBridgeService()
