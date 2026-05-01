from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from app.schemas.scripts import (
    RunningScript,
    ScriptLogResponse,
    ScriptRunRequest,
    ScriptRunResponse,
    ScriptStopResponse,
    ScriptRunStatus,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


class ScriptService:
    """
    Jetson-side script runner with:
    - subprocess execution
    - running/completed/failed/stopped status
    - stdout/stderr capture
    - log retrieval by run_id
    """

    def __init__(self) -> None:
        self._running: Dict[str, RunningScript] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._logs: Dict[str, ScriptLogResponse] = {}

        base = Path(__file__).resolve().parents[2] / "scripts"
        self._script_dir = base

        self._script_map = {
            "start_ros": self._script_dir / "start_ros.sh",
            "restart_roborio": self._script_dir / "restart_roborio.sh",
            "restart_jetson": self._script_dir / "restart_jetson.sh",
            "run_comm_check": self._script_dir / "run_comm_check.sh",
            "restart_camera": self._script_dir / "restart_camera.sh",
            "capture_picture": self._script_dir / "capture_picture.sh",
            "capture_video": self._script_dir / "capture_video.sh",
            "test_call": self._script_dir / "test_call.sh",
        }

    def _watch_process(self, run_id: str, proc: subprocess.Popen) -> None:
        try:
            stdout_data, stderr_data = proc.communicate()
            exit_code = proc.returncode
        except Exception as e:
            stdout_data = ""
            stderr_data = f"Error reading process output: {e}"
            exit_code = -1

        running = self._running.pop(run_id, None)
        self._procs.pop(run_id, None)

        if run_id not in self._logs:
            return

        log = self._logs[run_id]
        log.stdout = stdout_data or ""
        log.stderr = stderr_data or ""
        log.exit_code = exit_code
        log.finished_ms = _now_ms()

        if log.status == "stopped":
            # preserve stopped state if user stopped it
            return

        if exit_code == 0:
            log.status = "completed"
        else:
            log.status = "failed"

    def _cleanup_finished(self) -> None:
        finished = []
        for run_id, proc in self._procs.items():
            if proc.poll() is not None:
                finished.append(run_id)

        for run_id in finished:
            self._procs.pop(run_id, None)
            self._running.pop(run_id, None)

    def list_running(self) -> List[RunningScript]:
        self._cleanup_finished()
        return list(self._running.values())

    def run(self, req: ScriptRunRequest) -> ScriptRunResponse:
        self._cleanup_finished()

        script_path = self._script_map.get(req.name)
        if script_path is None:
            raise ValueError(f"Unknown script '{req.name}'")
        if not script_path.exists():
            raise ValueError(f"Script file not found: {script_path}")

        run_id = str(uuid.uuid4())
        started_ms = _now_ms()

        cmd = [str(script_path)]
        if req.args:
            for key, value in req.args.items():
                cmd.append(f"--{key}")
                cmd.append(str(value))

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,
            )
        except Exception as e:
            raise ValueError(f"Failed to start script '{req.name}': {e}")

        running = RunningScript(
            run_id=run_id,
            name=req.name,
            started_ms=started_ms,
            args=req.args,
            status="running",
            pid=proc.pid,
        )

        self._running[run_id] = running
        self._procs[run_id] = proc
        self._logs[run_id] = ScriptLogResponse(
            run_id=run_id,
            name=req.name,
            status="running",
            started_ms=started_ms,
            finished_ms=None,
            exit_code=None,
            stdout="",
            stderr="",
        )

        watcher = threading.Thread(
            target=self._watch_process,
            args=(run_id, proc),
            daemon=True,
        )
        watcher.start()

        return ScriptRunResponse(
            ok=True,
            run_id=run_id,
            started_ms=started_ms,
            running=self.list_running(),
        )

    def stop(self, run_id: str) -> ScriptStopResponse:
        self._cleanup_finished()

        proc = self._procs.get(run_id)
        log = self._logs.get(run_id)

        if proc is None or log is None:
            raise ValueError("run_id not found")

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            proc.terminate()

        log.status = "stopped"
        log.finished_ms = _now_ms()
        log.exit_code = None

        self._procs.pop(run_id, None)
        self._running.pop(run_id, None)

        return ScriptStopResponse(ok=True, run_id=run_id)

    def get_logs(self, run_id: str) -> ScriptLogResponse:
        log = self._logs.get(run_id)
        if log is None:
            raise ValueError("run_id not found")
        return log


script_service = ScriptService()
