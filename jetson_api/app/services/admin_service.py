from __future__ import annotations

import subprocess

from app.core.settings import settings
from app.schemas.admin import AdminActionResponse


class AdminService:
    def restart_service(self, service_name: str) -> AdminActionResponse:
        if not service_name.strip():
            raise ValueError("service_name is required")

        if settings.dry_run:
            return AdminActionResponse(
                ok=True,
                message=f"[dry_run] Would restart service '{service_name}'",
            )

        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", service_name],
                check=True,
                capture_output=True,
                text=True,
            )
            return AdminActionResponse(
                ok=True,
                message=f"Service '{service_name}' restarted",
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(e.stderr.strip() or f"Failed to restart service '{service_name}'")

    def restart_jetson(self) -> AdminActionResponse:
        if settings.dry_run or not settings.allow_reboot:
            return AdminActionResponse(
                ok=True,
                message="[dry_run] Would reboot Jetson",
            )

        try:
            subprocess.Popen(["sudo", "reboot"])
            return AdminActionResponse(
                ok=True,
                message="Jetson reboot initiated",
            )
        except Exception as e:
            raise ValueError(f"Failed to reboot Jetson: {e}")


admin_service = AdminService()