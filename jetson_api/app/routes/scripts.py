from typing import List
from fastapi import APIRouter, HTTPException

from app.schemas.scripts import (
    ScriptRunRequest,
    ScriptRunResponse,
    RunningScript,
    ScriptStopResponse,
    ScriptLogResponse,
)
from app.services.script_service import script_service

router = APIRouter(prefix="/scripts", tags=["scripts"])


@router.post("/run", response_model=ScriptRunResponse)
def run_script(req: ScriptRunRequest) -> ScriptRunResponse:
    try:
        return script_service.run(req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/running", response_model=List[RunningScript])
def list_running() -> List[RunningScript]:
    return script_service.list_running()


@router.post("/stop/{run_id}", response_model=ScriptStopResponse)
def stop_script(run_id: str) -> ScriptStopResponse:
    try:
        return script_service.stop(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="run_id not found")


@router.get("/{run_id}/logs", response_model=ScriptLogResponse)
def get_script_logs(run_id: str) -> ScriptLogResponse:
    try:
        return script_service.get_logs(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="run_id not found")