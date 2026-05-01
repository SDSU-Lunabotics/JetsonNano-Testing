from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


class MotorId(str, Enum):
    left_front = "left_front"
    right_front = "right_front"
    left_rear = "left_rear"
    right_rear = "right_rear"
    excavator = "excavator"
    deposition = "deposition"


class ActuatorId(str, Enum):
    excavator_1_actuator = "excavator_1_actuator"
    excavator_2_actuator = "excavator_2_actuator"
    excavator_3_actuator = "excavator_3_actuator"
    gate_actuator = "gate_actuator"
    camera_actuator = "camera_actuator"


ControlMode = Literal["manual", "autonomy"]


class Heartbeat(BaseModel):
    connected: bool
    last_seen_ms: Optional[int] = None
    age_ms: Optional[int] = None


class LinkStats(BaseModel):
    latency_ms: Optional[float] = None
    packet_loss_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    throughput_mbps: Optional[float] = None


class BatteryStatus(BaseModel):
    voltage_v: Optional[float] = None
    low: Optional[bool] = None
    source_key: Optional[str] = None
    raw_value: Optional[Any] = None
    bridge_connected: Optional[bool] = None


class ControllerInput(str, Enum):
    automation = "automation"
    excavator = "excavator"
    deposition = "deposition"
    move = "move"
    increase_speed = "increase_speed"
    decrease_speed = "decrease_speed"


class ControllerStatus(BaseModel):
    connected: bool
    last_input_ms: Optional[int] = None
    last_input: Optional[ControllerInput] = None


FaultSeverity = Literal["info", "warn", "error", "critical"]
FaultSource = Literal["jetson", "roborio", "network", "camera", "backend"]


class FaultCode(str, Enum):
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    NETWORK_HIGH_LATENCY = "NETWORK_HIGH_LATENCY"
    NETWORK_HIGH_PACKET_LOSS = "NETWORK_HIGH_PACKET_LOSS"
    BATTERY_LOW = "BATTERY_LOW"
    CAMERA_OFFLINE = "CAMERA_OFFLINE"
    ROBORIO_OFFLINE = "ROBORIO_OFFLINE"
    JETSON_OFFLINE = "JETSON_OFFLINE"
    MOTOR_OVERCURRENT = "MOTOR_OVERCURRENT"
    MOTOR_OVERTEMPERATURE = "MOTOR_OVERTEMPERATURE"
    MOTOR_TORQUE_FAULT = "MOTOR_TORQUE_FAULT"
    ACTUATOR_FAULT = "ACTUATOR_FAULT"


class Fault(BaseModel):
    code: FaultCode
    severity: FaultSeverity
    message: str
    source: Optional[FaultSource] = None
    timestamp_ms: int
