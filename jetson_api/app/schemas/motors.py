from __future__ import annotations

from typing import Any, Dict, Literal, Optional, List
from pydantic import BaseModel, Field
from .common import Fault, MotorId


class MotorHealth(BaseModel):
    motor_id: MotorId
    connected: Optional[bool] = None
    enabled: Optional[bool] = None
    active: Optional[bool] = None
    healthy: Optional[bool] = None
    faults: Optional[List[Fault]] = None
    fault: Optional[bool] = None
    fault_code: Optional[str] = None
    current_a: Optional[float] = None
    torque_nm: Optional[float] = None
    rpm: Optional[float] = None
    voltage_v: Optional[float] = None
    temperature_c: Optional[float] = None
    position_rotations: Optional[float] = None
    output_percent: Optional[float] = None
    target_rpm: Optional[float] = None
    target_percent: Optional[float] = None
    controller_mode: Optional[str] = None
    state: Optional[str] = None
    timestamp_ms: Optional[int] = None
    last_update_ms: Optional[int] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class MotorWarning(BaseModel):
    key: str
    value: Optional[float] = None
    limit: Optional[float] = None


class DriveMotorTelemetry(BaseModel):
    velocity_tps: Optional[float] = None
    torque_current_a: Optional[float] = None
    enabled: Optional[bool] = None
    applied_output: Optional[float] = None


class DriveTelemetry(BaseModel):
    left_front: DriveMotorTelemetry = Field(default_factory=DriveMotorTelemetry)
    left_rear: DriveMotorTelemetry = Field(default_factory=DriveMotorTelemetry)
    right_front: DriveMotorTelemetry = Field(default_factory=DriveMotorTelemetry)
    right_rear: DriveMotorTelemetry = Field(default_factory=DriveMotorTelemetry)
    left_output: Optional[float] = None
    right_output: Optional[float] = None
    active: Optional[bool] = None
    wheel_test_step: Optional[float] = None
    wheel_test_mode: Optional[str] = None


class ExcavationTargetsTelemetry(BaseModel):
    stow_bottom_counts: Optional[float] = None
    dig_zone_bottom_counts: Optional[float] = None


class ExcavationTelemetry(BaseModel):
    belt_running: Optional[bool] = None
    belt_output: Optional[float] = None
    belt_leader_torque_current_a: Optional[float] = None
    belt_follower_torque_current_a: Optional[float] = None
    left_extension_inches: Optional[float] = None
    right_extension_inches: Optional[float] = None
    left_extension_pct: Optional[float] = None
    right_extension_pct: Optional[float] = None
    position_calibrated: Optional[bool] = None
    bottom_left_counts: Optional[float] = None
    bottom_right_counts: Optional[float] = None
    bottom_left_inches: Optional[float] = None
    bottom_right_inches: Optional[float] = None
    bottom_left_extension_pct: Optional[float] = None
    bottom_right_extension_pct: Optional[float] = None
    jetson_left_extension_inches: Optional[float] = None
    jetson_right_extension_inches: Optional[float] = None
    jetson_left_extension_pct: Optional[float] = None
    jetson_right_extension_pct: Optional[float] = None
    jetson_bottom_position_calibrated: Optional[bool] = None
    bottom_diff_counts: Optional[float] = None
    sync_fault: Optional[bool] = None
    manual_bypass_sync: Optional[bool] = None
    bottom_position_calibrated: Optional[bool] = None
    position_reference: Optional[str] = None
    state: Optional[str] = None
    targets: ExcavationTargetsTelemetry = Field(default_factory=ExcavationTargetsTelemetry)


class DepositionTelemetry(BaseModel):
    running: Optional[bool] = None
    output: Optional[float] = None
    collecting_assist: Optional[bool] = None
    depositing: Optional[bool] = None
    door_state: Optional[str] = None
    torque_current_a: Optional[float] = None
    tailgate_counts: Optional[float] = None
    tailgate_inches: Optional[float] = None
    tailgate_extension_pct: Optional[float] = None
    tailgate_position_calibrated: Optional[bool] = None
    tailgate_state: Optional[str] = None
    tailgate_moving: Optional[bool] = None
    tailgate_open: Optional[bool] = None
    tailgate_closed: Optional[bool] = None
    tailgate_torque_current_a: Optional[float] = None


class CurrentLimitsTelemetry(BaseModel):
    enabled: Optional[bool] = None
    drive_a: Optional[float] = None
    excav_a: Optional[float] = None
    deposit_a: Optional[float] = None
    applied: Optional[bool] = None
    applied_drive_a: Optional[float] = None
    applied_excav_a: Optional[float] = None
    applied_deposit_a: Optional[float] = None


class JetsonMotorTelemetry(BaseModel):
    drive: DriveTelemetry = Field(default_factory=DriveTelemetry)
    excavation: ExcavationTelemetry = Field(default_factory=ExcavationTelemetry)
    deposition: DepositionTelemetry = Field(default_factory=DepositionTelemetry)
    current_limits: CurrentLimitsTelemetry = Field(default_factory=CurrentLimitsTelemetry)


class MotorCardStatus(BaseModel):
    motor_id: MotorId
    label: str
    active: Optional[bool] = None
    enabled: Optional[bool] = None
    current_a: Optional[float] = None
    output_percent: Optional[float] = None
    state: Optional[str] = None
    source: Optional[str] = None


class MotorCardsTelemetry(BaseModel):
    drive: MotorCardStatus
    left_front: MotorCardStatus
    left_rear: MotorCardStatus
    right_front: MotorCardStatus
    right_rear: MotorCardStatus
    excavator: MotorCardStatus
    deposition: MotorCardStatus
    conveyor: MotorCardStatus


class MotorsStatusResponse(BaseModel):
    timestamp_ms: int
    motors: List[MotorHealth]
    telemetry: JetsonMotorTelemetry = Field(default_factory=JetsonMotorTelemetry)
    cards: MotorCardsTelemetry
    raw_values: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[MotorWarning] = Field(default_factory=list)


MotorCommandMode = Literal["percent", "rpm", "stop"]


class MotorCommandRequest(BaseModel):
    mode: MotorCommandMode
    value: Optional[float] = None
    duration_ms: Optional[int] = Field(default=None, ge=0)


class MotorCommandResponse(BaseModel):
    ok: bool
    motor_id: str
    mode: Optional[MotorCommandMode] = None
    value: Optional[float] = None
    duration_ms: Optional[int] = None
    request_id: Optional[float] = None
    timestamp_ms: int
