from __future__ import annotations

import subprocess
from pathlib import Path

from app.core.settings import settings
from app.schemas.admin import AdminActionResponse


ROBORIO_RESTART_COMMAND = ["sudo", "-n", "/usr/local/bin/restart_roborio.sh"]


class AdminService:
    def __init__(self) -> None:
        self._script_dir = Path(__file__).resolve().parents[2] / "scripts"

    def restart_service(self, service_name: str) -> AdminActionResponse:
        if not service_name.strip():
            raise ValueError("service_name is required")

        if service_name == "restart_roborio":
            try:
                print(
                    f"[Jetson API] restart-service '{service_name}' launching {' '.join(ROBORIO_RESTART_COMMAND)}",
                    flush=True,
                )
                completed = subprocess.run(
                    ROBORIO_RESTART_COMMAND,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                message = completed.stdout.strip() or completed.stderr.strip() or "Roborio restart command completed"
                return AdminActionResponse(
                    ok=True,
                    message=message,
                )
            except subprocess.TimeoutExpired:
                raise ValueError("Roborio restart command timed out")
            except subprocess.CalledProcessError as e:
                message = e.stderr.strip() or e.stdout.strip() or f"exit code {e.returncode}"
                raise ValueError(f"Roborio restart command failed: {message}")
            except OSError as e:
                raise ValueError(f"Failed to run Roborio restart command: {e}")

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

        script_path = self._script_dir / "restart_jetson.sh"
        if not script_path.exists():
            raise ValueError(f"Restart script not found: {script_path}")

        try:
            subprocess.Popen([str(script_path)])
            return AdminActionResponse(
                ok=True,
                message="Jetson reboot initiated via restart_jetson.sh",
            )
        except Exception as e:
            raise ValueError(f"Failed to reboot Jetson: {e}")


admin_service = AdminService()
