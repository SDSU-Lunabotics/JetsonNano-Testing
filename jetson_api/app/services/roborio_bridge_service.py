from __future__ import annotations

import requests
from typing import Any, Dict


class RoboRIOBridgeService:
    def __init__(self) -> None:
        self._base_url = "http://127.0.0.1:8001"

    def get_status(self) -> Dict[str, Any]:
        try:
            r = requests.get(f"{self._base_url}/status", timeout=2)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return {
                "status": "error",
                "timestamp_ms": 0,
                "connected": False,
                "values": {},
                "warnings": [],
            }

    def is_connected(self) -> bool:
        payload = self.get_status()
        return bool(payload.get("connected", False))

    def get_value(self, key: str, default: Any = None) -> Any:
        payload = self.get_status()
        return payload.get("values", {}).get(key, default)


roborio_bridge_service = RoboRIOBridgeService()