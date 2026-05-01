import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from networktables import NetworkTables


HOST = "0.0.0.0"
PORT = 8001

# Make sure this matches your real RoboRIO IP
ROBORIO_IP = "10.0.9.2"

TORQUE_CURRENT_LIMIT_AMPS = 60.0
HEARTBEAT_STALE_MS = 2500

heartbeat_lock = threading.Lock()
heartbeat_state = {
    "last_value_sec": None,
    "last_seen_ms": None,
}

BATTERY_VALUE_KEYS = (
    "Battery/Voltage",
    "BatteryVoltage",
    "Battery Voltage",
    "RoboRIO/BatteryVoltage",
    "RoboRIO/Battery/Voltage",
    "Robot/BatteryVoltage",
    "RobotController/BatteryVoltage",
)


DATA_KEYS = {
    "Jetson/DriveForward": "number",
    "Jetson/DriveTurn": "number",
    "Jetson/DriveTimestamp": "number",
    "Jetson/WaypointRequested": "boolean",
    "Jetson/AutomationEnabled": "boolean",
    "Jetson/CommandReady": "boolean",
    "Jetson/CommandDuration": "number",
    "Jetson/CommandForward": "number",
    "Jetson/CommandTurn": "number",
    "Jetson/Command": "string",
    "Jetson/Speed": "number",
    "Jetson/TurnSpeed": "number",
    "Jetson/LedOverride": "boolean",
    "Jetson/LedState": "boolean",
    "Jetson/EStop": "boolean",
    "Jetson/ExcavatorEnabled": "boolean",
    "Jetson/ConveyorEnabled": "boolean",
    "Jetson/DepositionEnabled": "boolean",
    "NavX/YawDeg": "number",
    "Rover/HeartbeatSec": "number",
    "Rover/Connected": "boolean",
    "RoboRIO/HeartbeatSec": "number",
    "RoboRIO/Connected": "boolean",
    "RoboRIO/DriverStationAttached": "boolean",
    "RoboRIO/Enabled": "boolean",
    "RoboRIO/Mode": "string",
    "Controller/Connected": "boolean",
    "Controller/IsXbox": "boolean",
    "Controller/Present": "boolean",
    "Controller/ExpectedPortConnected": "boolean",
    "Controller/ExpectedPortIsXbox": "boolean",
    "Controller/Port": "number",
    "Controller/DetectedPort": "number",
    "Xbox/Connected": "boolean",
    "Xbox/Port": "number",
    "LEDEnabled": "boolean",
    "LEDState": "boolean",
    "LEDButton": "boolean",
    "ServoSweeping": "boolean",
    "ServoAngle": "number",
    "Kraken/LeftFront/VelocityTps": "number",
    "Kraken/LeftFront/TorqueCurrentA": "number",
    "Kraken/LeftRear/VelocityTps": "number",
    "Kraken/LeftRear/TorqueCurrentA": "number",
    "Kraken/RightFront/VelocityTps": "number",
    "Kraken/RightFront/TorqueCurrentA": "number",
    "Kraken/RightRear/VelocityTps": "number",
    "Kraken/RightRear/TorqueCurrentA": "number",
    "Battery/Voltage": "number",

    "Kraken/LeftFront/Enabled": "boolean",
    "Kraken/LeftRear/Enabled": "boolean",
    "Kraken/RightFront/Enabled": "boolean",
    "Kraken/RightRear/Enabled": "boolean",

    "Kraken/LeftFront/AppliedOutput": "number",
    "Kraken/LeftRear/AppliedOutput": "number",
    "Kraken/RightFront/AppliedOutput": "number",
    "Kraken/RightRear/AppliedOutput": "number",
    "Jetson/DriveLeftOutput": "number",
    "Jetson/DriveRightOutput": "number",
    "Jetson/DriveActive": "boolean",
    "Kraken/WheelTestStep": "number",
    "Kraken/WheelTestMode": "string",
    "Excav/BeltRunning": "boolean",
    "Excav/BeltOutput": "number",
    "Excav/BeltLeaderTorqueCurrentA": "number",
    "Excav/BeltFollowerTorqueCurrentA": "number",
    "Excav/BotLeftCounts": "number",
    "Excav/BotRightCounts": "number",
    "Excav/BottomDiffCounts": "number",
    "Excav/SyncFault": "boolean",
    "Excav/ManualBypassSync": "boolean",
    "Excav/BottomPositionCalibrated": "boolean",
    "Excav/PositionReference": "string",
    "Excav/State": "string",
    "Excav/Targets/StowBottomCounts": "number",
    "Excav/Targets/DigZoneBottomCounts": "number",
    "Deposit/Running": "boolean",
    "Deposit/Output": "number",
    "Deposit/CollectingAssist": "boolean",
    "Deposit/Depositing": "boolean",
    "Deposit/DoorState": "string",
    "Deposit/TorqueCurrentA": "number",
    "MainRover/CurrentLimitEnabled": "boolean",
    "MainRover/DriveCurrentLimitA": "number",
    "MainRover/ExcavCurrentLimitA": "number",
    "MainRover/DepositCurrentLimitA": "number",
    "MainRover/CurrentLimitApplied": "boolean",
    "MainRover/DriveCurrentLimitAppliedA": "number",
    "MainRover/ExcavCurrentLimitAppliedA": "number",
    "MainRover/DepositCurrentLimitAppliedA": "number",
}

DIRECT_READ_KEYS = {
    key
    for key in tuple(DATA_KEYS) + BATTERY_VALUE_KEYS
    if "/" in key
}


WRITE_KEYS = {
    "Jetson/LedOverride": "boolean",
    "Jetson/LedState": "boolean",
    "Jetson/EStop": "boolean",
    "Jetson/ExcavatorEnabled": "boolean",
    "Jetson/ConveyorEnabled": "boolean",
    "Jetson/DepositionEnabled": "boolean",
    "Jetson/AutomationEnabled": "boolean",
    "Jetson/CommandReady": "boolean",
    "Jetson/CommandDuration": "number",
    "Jetson/CommandForward": "number",
    "Jetson/CommandTurn": "number",
    "Jetson/Command": "string",
    "Jetson/DriveForward": "number",
    "Jetson/DriveTurn": "number",
    "Jetson/DriveTimestamp": "number",
    "Jetson/Speed": "number",
    "Jetson/TurnSpeed": "number",
}


print(f"Connecting to RoboRIO at {ROBORIO_IP}...")
NetworkTables.initialize(server=ROBORIO_IP)

sd = NetworkTables.getTable("SmartDashboard")


def now_ms():
    return int(time.time() * 1000)


def wait_for_connection(timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if NetworkTables.isConnected():
            return True
        time.sleep(0.1)
    return NetworkTables.isConnected()


def read_value(key, value_type, available_keys=None):
    try:
        if (
            available_keys is not None
            and key not in available_keys
        ):
            if key not in DIRECT_READ_KEYS:
                return None

            try:
                return sd.getValue(key)
            except Exception:
                return None

        if value_type == "boolean":
            return sd.getBoolean(key, False)
        if value_type == "number":
            return sd.getNumber(key, 0.0)
        return sd.getString(key, "")
    except Exception:
        return None


def parse_value(value_type, value):
    if value_type == "boolean":
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False

        return False

    if value_type == "number":
        try:
            return float(value)
        except Exception:
            return 0.0

    return str(value)


def write_value(key, value_type, value):
    parsed = parse_value(value_type, value)

    if value_type == "boolean":
        sd.putBoolean(key, parsed)
    elif value_type == "number":
        sd.putNumber(key, parsed)
    else:
        sd.putString(key, parsed)

    return parsed


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_available_keys():
    try:
        return set(sd.getKeys())
    except Exception:
        return set()


def read_battery_values(available_keys):
    values = {}

    for key in BATTERY_VALUE_KEYS:
        if key in DATA_KEYS:
            continue
        is_slash_key = "/" in key
        if available_keys and key not in available_keys and not is_slash_key:
            continue

        value = read_value(key, "number", available_keys)
        if value is not None:
            values[key] = value

    for key in available_keys:
        normalized = re.sub(r"[^a-z0-9]+", "", key.lower())
        if "battery" not in normalized or "voltage" not in normalized:
            continue
        if key in DATA_KEYS or key in values:
            continue

        value = read_value(key, "number", available_keys)
        if value is not None:
            values[key] = value

    return values


def normalize_battery_voltage(values):
    canonical_key = "Battery/Voltage"
    if _safe_float(values.get(canonical_key)) is not None:
        return {
            "source_key": canonical_key,
            "voltage_v": float(values[canonical_key]),
            "aliases": {},
        }

    aliases = {
        key: value
        for key, value in values.items()
        if key != canonical_key
        and (
            key in BATTERY_VALUE_KEYS
            or (
                "battery" in re.sub(r"[^a-z0-9]+", "", key.lower())
                and "voltage" in re.sub(r"[^a-z0-9]+", "", key.lower())
            )
        )
        and _safe_float(value) is not None
    }

    for key in BATTERY_VALUE_KEYS:
        value = aliases.get(key)
        voltage = _safe_float(value)
        if voltage is not None:
            values[canonical_key] = voltage
            return {
                "source_key": key,
                "voltage_v": voltage,
                "aliases": aliases,
            }

    for key, value in aliases.items():
        voltage = _safe_float(value)
        if voltage is not None:
            values[canonical_key] = voltage
            return {
                "source_key": key,
                "voltage_v": voltage,
                "aliases": aliases,
            }

    return {
        "source_key": None,
        "voltage_v": None,
        "aliases": aliases,
    }


def torque_warnings(values):
    warnings = []

    for key, value_type in DATA_KEYS.items():
        if not key.endswith("/TorqueCurrentA"):
            continue

        value = _safe_float(values.get(key))
        if value is None:
            continue

        if value_type == "number" and value > TORQUE_CURRENT_LIMIT_AMPS:
            warnings.append(
                {
                    "key": key,
                    "value": value,
                    "limit": TORQUE_CURRENT_LIMIT_AMPS,
                }
            )

    return warnings


def update_heartbeat_status(values):
    raw_value = values.get("RoboRIO/HeartbeatSec")
    now = now_ms()

    try:
        value_sec = float(raw_value)
    except (TypeError, ValueError):
        value_sec = None

    with heartbeat_lock:
        if value_sec is not None:
            previous = heartbeat_state["last_value_sec"]
            if previous is None or value_sec != previous:
                heartbeat_state["last_value_sec"] = value_sec
                heartbeat_state["last_seen_ms"] = now

        last_seen_ms = heartbeat_state["last_seen_ms"]

    age_ms = None if last_seen_ms is None else now - last_seen_ms

    return {
        "present": value_sec is not None,
        "value_sec": value_sec,
        "last_seen_ms": last_seen_ms,
        "age_ms": age_ms,
        "fresh": age_ms is not None and age_ms < HEARTBEAT_STALE_MS,
        "stale_after_ms": HEARTBEAT_STALE_MS,
    }


def heartbeat_warnings(connected, heartbeat):
    if not connected:
        return []

    if not heartbeat["present"]:
        return [
            {
                "source": "roborio",
                "key": "RoboRIO/HeartbeatSec",
                "message": "NetworkTables is connected, but the RoboRIO heartbeat key is missing.",
            }
        ]

    if not heartbeat["fresh"]:
        return [
            {
                "source": "roborio",
                "key": "RoboRIO/HeartbeatSec",
                "age_ms": heartbeat["age_ms"],
                "limit_ms": HEARTBEAT_STALE_MS,
                "message": "RoboRIO heartbeat key is present, but it is not changing.",
            }
        ]

    return []


def read_dynamic_controller_values():
    """
    Pull controller-related SmartDashboard entries even if they are not part of the
    fixed DATA_KEYS whitelist. This lets the Jetson status API surface Xbox state
    when the RoboRIO publishes it under controller-ish keys.
    """
    try:
        keys = sd.getKeys()
    except Exception:
        return {}

    controller_tokens = ("controller", "xbox", "joystick", "gamepad")
    values = {}

    for key in keys:
        normalized = key.lower()
        if not any(token in normalized for token in controller_tokens):
            continue
        if key in DATA_KEYS:
            continue

        try:
            values[key] = sd.getValue(key)
        except Exception:
            continue

    return values


def infer_controller_status(values):
    controller_keys = {}
    connected = None
    is_xbox = None
    expected_port = None
    detected_port = None
    last_input_ms = None
    last_input = None

    for key, value in values.items():
        normalized = key.lower()

        if any(token in normalized for token in ("controller", "xbox", "joystick", "gamepad")):
            controller_keys[key] = value

    connected_candidates = []
    for key in (
        "Controller/Connected",
        "Controller/Present",
        "Xbox/Connected",
        "Controller/ExpectedPortConnected",
    ):
        value = controller_keys.get(key)
        if value is not None:
            connected_candidates.append(parse_value("boolean", value))

    if connected_candidates:
        connected = any(connected_candidates)

    xbox_candidates = []
    for key in ("Controller/IsXbox", "Controller/ExpectedPortIsXbox"):
        value = controller_keys.get(key)
        if value is not None:
            xbox_candidates.append(parse_value("boolean", value))

    if xbox_candidates:
        is_xbox = any(xbox_candidates)

    expected_port = controller_keys.get("Controller/Port")
    detected_port = controller_keys.get("Controller/DetectedPort", controller_keys.get("Xbox/Port"))

    for key, value in controller_keys.items():
        normalized = key.lower()

        if any(token in normalized for token in ("connected", "present", "detected", "plugged")):
            candidate = parse_value("boolean", value)
            if connected is None:
                connected = candidate
            else:
                connected = connected or candidate
            continue

        if last_input_ms is None and any(token in normalized for token in ("lastinputms", "last_input_ms", "lastinput", "last_input", "timestamp")):
            try:
                numeric = int(float(value))
            except Exception:
                numeric = None

            if numeric is not None and numeric > 0:
                last_input_ms = numeric
            continue

        if last_input is None and any(token in normalized for token in ("lastaction", "last_action", "lastinputname", "last_input_name", "input", "action")):
            if value is not None:
                text = str(value).strip()
                if text:
                    last_input = text

    if connected is None and last_input_ms is not None:
        connected = (now_ms() - last_input_ms) < 5000

    return {
        "connected": bool(connected) if connected is not None else False,
        "is_xbox": bool(is_xbox) if is_xbox is not None else False,
        "expected_port": (
            int(float(expected_port)) if expected_port is not None else None
        ),
        "detected_port": (
            int(float(detected_port))
            if detected_port is not None and float(detected_port) >= 0
            else None
        ),
        "last_input_ms": last_input_ms,
        "last_input": last_input,
        "source_keys": sorted(controller_keys.keys()),
    }


def _float_arg(data, key, default):
    try:
        return float(data.get(key, default))
    except Exception:
        return float(default)


def _bool_arg(data, key, default):
    return parse_value("boolean", data.get(key, default))


def _str_arg(data, key, default=""):
    value = data.get(key, default)
    if value is None:
        return str(default)
    return str(value).strip()


def _set_deposition_enabled(enabled):
    write_value("Jetson/ConveyorEnabled", "boolean", enabled)
    write_value("Jetson/DepositionEnabled", "boolean", enabled)


def _handle_motor_command(motor_id, data):
    normalized_motor_id = str(motor_id).strip().lower().replace("-", "_").replace(" ", "_")
    mode = _str_arg(data, "mode", "stop").lower()
    value = data.get("value")
    duration_ms = data.get("duration_ms")
    timestamp_ms = now_ms()

    if mode not in {"percent", "rpm", "stop"}:
        return 400, {
            "status": "error",
            "message": f"Unsupported motor mode: {mode}",
        }

    if not wait_for_connection():
        return 503, {
            "status": "error",
            "message": "RoboRIO not connected",
        }

    if normalized_motor_id == "excavator":
        enabled = mode != "stop" and parse_value("number", value) != 0.0
        write_value("Jetson/ExcavatorEnabled", "boolean", enabled)
    elif normalized_motor_id in {"deposition", "conveyor"}:
        enabled = mode != "stop" and parse_value("number", value) != 0.0
        _set_deposition_enabled(enabled)
        normalized_motor_id = "deposition"
    else:
        return 400, {
            "status": "error",
            "message": f"Unsupported motor id: {motor_id}",
        }

    return 200, {
        "ok": True,
        "status": "ok",
        "timestamp_ms": timestamp_ms,
        "motor_id": normalized_motor_id,
        "mode": mode,
        "value": None if value is None else parse_value("number", value),
        "duration_ms": duration_ms,
    }


def stop_all_motion():
    write_value("Jetson/AutomationEnabled", "boolean", False)
    write_value("Jetson/CommandForward", "number", 0.0)
    write_value("Jetson/CommandTurn", "number", 0.0)
    write_value("Jetson/CommandDuration", "number", 0.0)
    write_value("Jetson/CommandReady", "boolean", False)
    write_value("Jetson/Command", "string", "")

    write_value("Jetson/DriveForward", "number", 0.0)
    write_value("Jetson/DriveTurn", "number", 0.0)
    write_value("Jetson/DriveTimestamp", "number", now_ms())


def run_motion_command(duration, speed, turn=0.0, command="drive_forward"):
    write_value("Jetson/AutomationEnabled", "boolean", True)
    write_value("Jetson/Command", "string", command)
    write_value("Jetson/CommandForward", "number", speed)
    write_value("Jetson/CommandTurn", "number", turn)
    write_value("Jetson/CommandDuration", "number", duration)
    write_value("Jetson/CommandReady", "boolean", True)

    write_value("Jetson/DriveForward", "number", speed)
    write_value("Jetson/DriveTurn", "number", turn)
    write_value("Jetson/DriveTimestamp", "number", now_ms())

    time.sleep(0.1)
    write_value("Jetson/CommandReady", "boolean", False)

    time.sleep(max(0.0, duration))
    stop_all_motion()


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/ping":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "timestamp_ms": now_ms(),
                },
            )
            return

        if self.path == "/keys":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "timestamp_ms": now_ms(),
                    "keys": [
                        {"key": key, "type": value_type}
                        for key, value_type in DATA_KEYS.items()
                    ],
                },
            )
            return

        if self.path == "/status":
            available_keys = read_available_keys()

            values = {
                key: read_value(key, value_type, available_keys)
                for key, value_type in DATA_KEYS.items()
            }
            dynamic_controller_values = read_dynamic_controller_values()
            values.update(dynamic_controller_values)
            values.update(read_battery_values(available_keys))
            battery = normalize_battery_voltage(values)
            connected = NetworkTables.isConnected()
            heartbeat = update_heartbeat_status(values)

            self._send_json(
                200,
                {
                    "status": "ok",
                    "timestamp_ms": now_ms(),
                    "connected": connected,
                    "heartbeat": heartbeat,
                    "values": values,
                    "battery": battery,
                    "controller": infer_controller_status(values),
                    "warnings": torque_warnings(values) + heartbeat_warnings(connected, heartbeat),
                },
            )
            return

        self._send_json(404, {"status": "error", "message": "Not found"})

    def do_POST(self):

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        motor_match = re.fullmatch(r"/motors/([^/]+)/command", self.path)
        if motor_match:
            status, payload = _handle_motor_command(motor_match.group(1), data)
            self._send_json(status, payload)
            return

        if self.path == "/set":

            key = data.get("key")

            if key not in WRITE_KEYS:
                self._send_json(
                    400,
                    {
                        "status": "error",
                        "message": "Key not writable",
                    },
                )
                return

            if not wait_for_connection():
                self._send_json(
                    503,
                    {
                        "status": "error",
                        "message": "RoboRIO not connected",
                    },
                )
                return

            value_type = WRITE_KEYS[key]
            value = write_value(key, value_type, data.get("value"))

            self._send_json(
                200,
                {
                    "status": "ok",
                    "timestamp_ms": now_ms(),
                    "key": key,
                    "type": value_type,
                    "value": value,
                },
            )
            return

        if self.path == "/led":

            led_state = _bool_arg(data, "state", False)
            override = _bool_arg(data, "override", True)

            if not wait_for_connection():
                self._send_json(
                    503,
                    {
                        "status": "error",
                        "message": "RoboRIO not connected",
                    },
                )
                return

            threading.Thread(
                target=write_value,
                args=("Jetson/LedOverride", "boolean", override),
                daemon=True,
            ).start()

            threading.Thread(
                target=write_value,
                args=("Jetson/LedState", "boolean", led_state),
                daemon=True,
            ).start()

            self._send_json(
                200,
                {
                    "status": "ok",
                    "timestamp_ms": now_ms(),
                    "led": led_state,
                    "override": override,
                },
            )
            return

        if self.path == "/drive/forward":
            duration = _float_arg(data, "duration", 3.0)
            speed = _float_arg(data, "speed", 0.6)

            if duration <= 0.0:
                self._send_json(400, {"status": "error", "message": "duration must be > 0"})
                return

            if not wait_for_connection():
                self._send_json(
                    503,
                    {"status": "error", "message": "RoboRIO not connected"},
                )
                return

            threading.Thread(
                target=run_motion_command,
                args=(duration, speed, 0.0, "drive_forward"),
                daemon=True,
            ).start()

            self._send_json(
                200,
                {
                    "status": "ok",
                    "timestamp_ms": now_ms(),
                    "action": "drive_forward",
                    "duration": duration,
                    "speed": speed,
                },
            )
            return

        if self.path == "/automation/start":
            duration = _float_arg(data, "duration", 3.0)
            speed = _float_arg(data, "speed", 0.6)
            turn = _float_arg(data, "turn", 0.0)
            command = str(data.get("command", "drive_forward"))

            if duration <= 0.0:
                self._send_json(400, {"status": "error", "message": "duration must be > 0"})
                return

            if not wait_for_connection():
                self._send_json(
                    503,
                    {"status": "error", "message": "RoboRIO not connected"},
                )
                return

            threading.Thread(
                target=run_motion_command,
                args=(duration, speed, turn, command),
                daemon=True,
            ).start()

            self._send_json(
                200,
                {
                    "status": "ok",
                    "timestamp_ms": now_ms(),
                    "action": "automation_start",
                    "command": command,
                    "duration": duration,
                    "speed": speed,
                    "turn": turn,
                },
            )
            return

        if self.path == "/automation/stop":
            if not wait_for_connection():
                self._send_json(
                    503,
                    {"status": "error", "message": "RoboRIO not connected"},
                )
                return

            stop_all_motion()
            self._send_json(
                200,
                {
                    "status": "ok",
                    "timestamp_ms": now_ms(),
                    "action": "automation_stop",
                },
            )
            return

        if self.path == "/control/estop":
            engage = _bool_arg(data, "engage", True)

            if not wait_for_connection():
                self._send_json(
                    503,
                    {"status": "error", "message": "RoboRIO not connected"},
                )
                return

            write_value("Jetson/EStop", "boolean", engage)
            if engage:
                stop_all_motion()
                write_value("Jetson/ExcavatorEnabled", "boolean", False)
                write_value("Jetson/ConveyorEnabled", "boolean", False)

            self._send_json(
                200,
                {
                    "status": "ok",
                    "timestamp_ms": now_ms(),
                    "estop": engage,
                },
            )
            return

        if self.path == "/actuators/excavator/start":
            if not wait_for_connection():
                self._send_json(
                    503,
                    {"status": "error", "message": "RoboRIO not connected"},
                )
                return
            write_value("Jetson/ExcavatorEnabled", "boolean", True)
            self._send_json(200, {"status": "ok", "timestamp_ms": now_ms(), "excavator": True})
            return

        if self.path == "/actuators/excavator/stop":
            if not wait_for_connection():
                self._send_json(
                    503,
                    {"status": "error", "message": "RoboRIO not connected"},
                )
                return
            write_value("Jetson/ExcavatorEnabled", "boolean", False)
            self._send_json(200, {"status": "ok", "timestamp_ms": now_ms(), "excavator": False})
            return

        if self.path == "/actuators/conveyor/start":
            if not wait_for_connection():
                self._send_json(
                    503,
                    {"status": "error", "message": "RoboRIO not connected"},
                )
                return
            _set_deposition_enabled(True)
            self._send_json(200, {"status": "ok", "timestamp_ms": now_ms(), "conveyor": True})
            return

        if self.path == "/actuators/conveyor/stop":
            if not wait_for_connection():
                self._send_json(
                    503,
                    {"status": "error", "message": "RoboRIO not connected"},
                )
                return
            _set_deposition_enabled(False)
            self._send_json(200, {"status": "ok", "timestamp_ms": now_ms(), "conveyor": False})
            return

        if self.path == "/actuators/deposition/start":
            if not wait_for_connection():
                self._send_json(
                    503,
                    {"status": "error", "message": "RoboRIO not connected"},
                )
                return
            _set_deposition_enabled(True)
            self._send_json(200, {"status": "ok", "timestamp_ms": now_ms(), "deposition": True})
            return

        if self.path == "/actuators/deposition/stop":
            if not wait_for_connection():
                self._send_json(
                    503,
                    {"status": "error", "message": "RoboRIO not connected"},
                )
                return
            _set_deposition_enabled(False)
            self._send_json(200, {"status": "ok", "timestamp_ms": now_ms(), "deposition": False})
            return

        self._send_json(404, {"status": "error", "message": "Not found"})

    def log_message(self, format, *args):
        return

    def _send_json(self, status, payload):

        resp_bytes = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()

        self.wfile.write(resp_bytes)


if __name__ == "__main__":

    print("Starting RoboRIO bridge...")
    print(f"Listening on http://{HOST}:{PORT}")

    server = ThreadingHTTPServer((HOST, PORT), Handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down RoboRIO bridge...")
        server.shutdown()
