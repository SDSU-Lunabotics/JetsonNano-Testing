from __future__ import annotations

from typing import Optional, Dict, Union, Literal, List
from pydantic import BaseModel


ScriptArg = Union[str, int, float, bool]


ScriptName = Literal[
    "start_ros",
    "restart_lidar",
    "restart_roborio",
    "restart_jetson",
    "run_comm_check",
    "restart_camera",
    "capture_picture",
    "capture_video",
]


ScriptRunStatus = Literal["running", "completed", "failed", "stopped"]


class RunningScript(BaseModel):
    run_id: str
    name: ScriptName
    started_ms: int
    args: Optional[Dict[str, ScriptArg]] = None
    status: ScriptRunStatus = "running"
    pid: Optional[int] = None


class ScriptRunRequest(BaseModel):
    name: ScriptName
    args: Optional[Dict[str, ScriptArg]] = None


class ScriptRunResponse(BaseModel):
    ok: bool
    run_id: str
    started_ms: int
    running: List[RunningScript] = []


class ScriptStopResponse(BaseModel):
    ok: bool
    run_id: str


class ScriptLogResponse(BaseModel):
    run_id: str
    name: ScriptName
    status: ScriptRunStatus
    started_ms: int
    finished_ms: Optional[int] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""