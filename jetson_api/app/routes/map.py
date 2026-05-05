import json
from pathlib import Path
import time
from typing import Iterator, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse

from app.core.settings import settings
from app.schemas.map import (
    DriveCalibrationState,
    MapFrameIngestResponse,
    MapStreamStatus,
    MapUiCommandRequest,
    MapUiCommandResponse,
    MapUiControl,
    MapUiStateResponse,
    MapWaypointClickRequest,
    MapWaypointCommandResponse,
)
from app.services.map_service import map_service

router = APIRouter(prefix="/map", tags=["map"])


_ONE_PX_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
    b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xcf \xff\xd9"
)

_SUPPORTED_UI_COMMANDS = {
    "paint_obstacle",
    "paint_safe",
    "draw_safe",
    "erase_safe",
    "clear_all",
    "clear_paint",
    "lock_green",
    "reset_map",
    "reset_confirm",
    "reset_cancel",
    "localize_scan",
    "direct_nav",
    "main_rover_mode",
    "camera_view",
    "set_control_mode",
    "auto_digger",
    "camera_overlay",
    "drive_heading_flip",
    "hard_drive_flip",
    "camera_view_flip",
    "display_heading_flip",
    "direct_nav",
    "drive_calibration_mode",
    "drive_calibration_cancel",
    "dig_style_cycle",
    "dig_phase_cycle",
    "dig_record_dig",
    "dig_record_retract",
    "dig_record_stop",
    "dig_profile_prev",
    "dig_profile_next",
    "dig_profile_use",
    "dig_profile_delete",
    "test_excavation_dig",
    "test_excavation_left_extend",
    "test_excavation_right_extend",
    "test_excavation_lower",
    "door_open",
    "door_close",
    "stop_actuators",
    "main_rover_mode",
    "camera_view",
    "draw_excav_zone",
    "draw_deposit_zone",
    "pick_dig_start",
    "brush_minus",
    "brush_plus",
    "set_brush_radius",
}

_COMMAND_ALIASES = {
    "draw_safe": "paint_safe",
    "clear_paint": "clear_all",
}


def _default_ui_controls() -> List[MapUiControl]:
    return [
        MapUiControl(id="paint_obstacle", label="Paint Obstacle", command="paint_obstacle"),
        MapUiControl(id="paint_safe", label="Paint Safe", command="paint_safe"),
        MapUiControl(id="erase_safe", label="Erase Safe", command="erase_safe"),
        MapUiControl(id="clear_all", label="Clear All", command="clear_all"),
        MapUiControl(id="draw_excav_zone", label="Draw Excav Zone", command="draw_excav_zone"),
        MapUiControl(id="draw_deposit_zone", label="Draw Deposit Zone", command="draw_deposit_zone"),
    ]


def _write_waypoint_command(command: dict) -> None:
    target = Path(settings.map_waypoint_command_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f"{target.suffix}.tmp")
    tmp.write_text(json.dumps(command), encoding="utf-8")
    tmp.replace(target)


def _command_seq() -> int:
    return int(time.time_ns() // 1000)


def _read_ui_state() -> MapUiStateResponse:
    target = Path(settings.map_ui_state_file)
    if not target.exists():
        return MapUiStateResponse(available=False, controls=_default_ui_controls())

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return MapUiStateResponse(available=False, controls=_default_ui_controls())

    controls_raw = payload.get("controls") or []
    controls: List[MapUiControl] = []
    for item in controls_raw:
        try:
            controls.append(MapUiControl(**item))
        except Exception:
            continue
    if not controls:
        controls = _default_ui_controls()

    drive_calibration = None
    drive_calibration_raw = payload.get("drive_calibration")
    if isinstance(drive_calibration_raw, dict):
        try:
            drive_calibration = DriveCalibrationState(**drive_calibration_raw)
        except Exception:
            drive_calibration = None

    return MapUiStateResponse(
        available=bool(payload.get("available", True)),
        source=payload.get("source"),
        timestamp_ms=payload.get("timestamp_ms"),
        mining_state=payload.get("mining_state"),
        localization_scan_active=payload.get("localization_scan_active"),
        landmark_count=payload.get("landmark_count"),
        selected_tool=payload.get("selected_tool"),
        brush_radius=payload.get("brush_radius"),
        brush_radius_min=payload.get("brush_radius_min"),
        brush_radius_max=payload.get("brush_radius_max"),
        drive_calibration=drive_calibration,
        controls=controls,
    )


@router.get("/status", response_model=MapStreamStatus)
def get_map_status() -> MapStreamStatus:
    return map_service.get_status()


@router.get("/ui", response_model=MapUiStateResponse)
def get_map_ui_state() -> MapUiStateResponse:
    return _read_ui_state()


@router.get("/snapshot.jpg")
def get_snapshot_jpeg() -> Response:
    jpeg = map_service.get_latest_jpeg_bytes() or _ONE_PX_JPEG
    return Response(content=jpeg, media_type="image/jpeg")


def _mjpeg_stream(frame_interval_ms: int) -> Iterator[bytes]:
    sleep_s = max(frame_interval_ms, 1) / 1000.0
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    last_frame = _ONE_PX_JPEG

    while True:
        frame = map_service.get_latest_jpeg_bytes() or last_frame
        if frame:
            last_frame = frame
        yield boundary + last_frame + b"\r\n"
        time.sleep(sleep_s)


@router.get("/stream")
def get_stream(
    frame_interval_ms: int = Query(default=settings.map_stream_interval_ms, ge=50, le=2000),
) -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_stream(frame_interval_ms),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/frame", response_model=MapFrameIngestResponse)
async def post_map_frame(
    request: Request,
    width: Optional[int] = Query(default=None, ge=1),
    height: Optional[int] = Query(default=None, ge=1),
    source: Optional[str] = Query(default=None),
    timestamp_ms: Optional[int] = Query(default=None, ge=0),
) -> MapFrameIngestResponse:
    payload = await request.body()
    frame_seq = map_service.ingest_jpeg(
        payload,
        width=width,
        height=height,
        source=source,
        source_timestamp_ms=timestamp_ms,
    )
    return MapFrameIngestResponse(
        ok=True,
        timestamp_ms=int(time.time() * 1000),
        frame_seq=frame_seq,
    )


@router.post("/waypoint", response_model=MapWaypointCommandResponse)
def post_waypoint(req: MapWaypointClickRequest) -> MapWaypointCommandResponse:
    command_seq = _command_seq()
    timestamp_ms = int(time.time() * 1000)
    _write_waypoint_command(
        {
            "seq": command_seq,
            "type": "set_goal_click",
            "display_x": int(req.display_x),
            "display_y": int(req.display_y),
            "source": req.source,
            "timestamp_ms": timestamp_ms,
        }
    )
    return MapWaypointCommandResponse(
        ok=True,
        timestamp_ms=timestamp_ms,
        command_seq=command_seq,
    )


@router.post("/command", response_model=MapUiCommandResponse)
def post_map_ui_command(req: MapUiCommandRequest) -> MapUiCommandResponse:
    command = str(req.command).strip()
    if command not in _SUPPORTED_UI_COMMANDS:
        raise HTTPException(status_code=400, detail=f"Unsupported map UI command: {command}")
    action = _COMMAND_ALIASES.get(command, command)

    payload = {
        "seq": _command_seq(),
        "type": "ui_action",
        "action": action,
        "source": req.source,
        "timestamp_ms": int(time.time() * 1000),
    }
    if action == "set_control_mode":
        if req.mode is None:
            raise HTTPException(status_code=400, detail="mode is required for set_control_mode")
        payload["mode"] = req.mode
    if req.value is not None:
        payload["value"] = int(req.value)

    command_seq = int(payload["seq"])
    _write_waypoint_command(payload)
    return MapUiCommandResponse(
        ok=True,
        timestamp_ms=int(payload["timestamp_ms"]),
        command_seq=command_seq,
        command=command,
    )
