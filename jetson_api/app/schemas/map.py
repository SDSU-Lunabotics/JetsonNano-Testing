from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.common import ControlMode


class MapStreamStatus(BaseModel):
    available: bool = False
    source: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    last_frame_ms: Optional[int] = None
    source_timestamp_ms: Optional[int] = None
    frame_seq: int = 0


class MapFrameIngestResponse(BaseModel):
    ok: bool
    timestamp_ms: int
    frame_seq: int


class MapWaypointClickRequest(BaseModel):
    display_x: int = Field(ge=0)
    display_y: int = Field(ge=0)
    source: Optional[str] = None


class MapWaypointCommandResponse(BaseModel):
    ok: bool
    timestamp_ms: int
    command_seq: int


class MapUiControl(BaseModel):
    id: str
    label: str
    command: str
    active: bool = False
    enabled: bool = True


class DriveCalibrationState(BaseModel):
    active: bool = False
    target_cell: Optional[List[int]] = None
    last_result: Optional[str] = None
    saved_drive_heading_flip: Optional[bool] = None
    saved_hard_drive_flip: Optional[bool] = None
    saved_display_heading_flip: Optional[bool] = None
    saved_camera_map_angle_deg: Optional[float] = None
    saved_camera_deposit_angle_deg: Optional[float] = None


class ActuatorUiState(BaseModel):
    left_extension_pct: Optional[float] = None
    right_extension_pct: Optional[float] = None
    left_extension_inches: Optional[float] = None
    right_extension_inches: Optional[float] = None
    tailgate_extension_pct: Optional[float] = None
    tailgate_inches: Optional[float] = None
    tailgate_counts: Optional[float] = None
    tailgate_position_calibrated: Optional[bool] = None
    tailgate_state: Optional[str] = None
    tailgate_moving: Optional[bool] = None
    tailgate_open: Optional[bool] = None
    tailgate_closed: Optional[bool] = None
    bottom_position_calibrated: Optional[bool] = None
    sync_fault: Optional[bool] = None
    left_extend_command: bool = False
    right_extend_command: bool = False
    dig_command: bool = False
    lower_command: bool = False
    door_open_command: bool = False
    door_close_command: bool = False


class MapUiStateResponse(BaseModel):
    available: bool = False
    source: Optional[str] = None
    timestamp_ms: Optional[int] = None
    mining_state: Optional[str] = None
    localization_scan_active: Optional[bool] = None
    landmark_count: Optional[int] = None
    selected_tool: Optional[str] = None
    brush_radius: Optional[int] = None
    brush_radius_min: Optional[int] = None
    brush_radius_max: Optional[int] = None
    drive_calibration: Optional[DriveCalibrationState] = None
    actuators: Optional[ActuatorUiState] = None
    controls: List[MapUiControl] = Field(default_factory=list)


class MapUiCommandRequest(BaseModel):
    command: str = Field(
        description=(
            "UI map action. Supported values include paint_obstacle, paint_safe, erase_safe, "
            "clear_all, lock_green, reset_map, reset_confirm, reset_cancel, localize_scan, "
            "direct_nav, main_rover_mode, camera_view, set_control_mode, auto_run, draw_excav_zone, "
            "draw_deposit_zone, set_starting_zone, lock_start_frame, pick_dig_start, brush_minus, brush_plus, set_brush_radius."
            "auto_digger, camera_overlay, drive_heading_flip, display_heading_flip, direct_nav, "
            "direct_nav, main_rover_mode, camera_view, set_control_mode, draw_excav_zone, "
            "draw_deposit_zone, set_starting_zone, lock_start_frame, set_berm_left, set_berm_right, pick_dig_start, "
            "brush_minus, brush_plus, set_brush_radius, "
            "auto_digger, camera_overlay, drive_heading_flip, hard_drive_flip, "
            "camera_view_flip, display_heading_flip, "
            "drive_calibration_mode, drive_calibration_cancel, dig_style_cycle, "
            "dig_phase_cycle, dig_record_dig, dig_record_retract, dig_record_stop, "
            "dig_profile_prev, dig_profile_next, dig_profile_use, dig_profile_delete, "
            "test_excavation_dig, "
            "test_excavation_left_extend, test_excavation_right_extend, "
            "test_excavation_lower, door_open, door_close, stop_actuators, "
            "main_rover_mode, camera_view, draw_excav_zone, draw_deposit_zone, set_starting_zone, lock_start_frame, "
            "set_berm_left, set_berm_right, pick_dig_start, brush_minus, "
            "brush_plus, set_brush_radius."
        )
    )
    source: Optional[str] = None
    mode: Optional[ControlMode] = None
    value: Optional[int] = None


class MapUiCommandResponse(BaseModel):
    ok: bool
    timestamp_ms: int
    command_seq: int
    command: str
