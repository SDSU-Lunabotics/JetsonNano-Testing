import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from networktables import NetworkTables


HOST = "0.0.0.0"
PORT = 8001

# Make sure this matches your real RoboRIO IP
ROBORIO_IP = "10.0.9.2"

TORQUE_CURRENT_LIMIT_AMPS = 60.0


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
    "NavX/YawDeg": "number",
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
}


WRITE_KEYS = {
    "Jetson/LedOverride": "boolean",
    "Jetson/LedState": "boolean",
    "Jetson/EStop": "boolean",
    "Jetson/ExcavatorEnabled": "boolean",
    "Jetson/ConveyorEnabled": "boolean",
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


def read_value(key, value_type):
    try:
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


def torque_warnings(values):
    warnings = []

    for key, value_type in DATA_KEYS.items():
        if not key.endswith("/TorqueCurrentA"):
            continue

        value = values.get(key, 0.0)

        if value_type == "number" and value > TORQUE_CURRENT_LIMIT_AMPS:
            warnings.append(
                {
                    "key": key,
                    "value": value,
                    "limit": TORQUE_CURRENT_LIMIT_AMPS,
                }
            )

    return warnings


def _float_arg(data, key, default):
    try:
        return float(data.get(key, default))
    except Exception:
        return float(default)


def _bool_arg(data, key, default):
    return parse_value("boolean", data.get(key, default))


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

            values = {
                key: read_value(key, value_type)
                for key, value_type in DATA_KEYS.items()
            }

            self._send_json(
                200,
                {
                    "status": "ok",
                    "timestamp_ms": now_ms(),
                    "connected": NetworkTables.isConnected(),
                    "values": values,
                    "warnings": torque_warnings(values),
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
            write_value("Jetson/ConveyorEnabled", "boolean", True)
            self._send_json(200, {"status": "ok", "timestamp_ms": now_ms(), "conveyor": True})
            return

        if self.path == "/actuators/conveyor/stop":
            if not wait_for_connection():
                self._send_json(
                    503,
                    {"status": "error", "message": "RoboRIO not connected"},
                )
                return
            write_value("Jetson/ConveyorEnabled", "boolean", False)
            self._send_json(200, {"status": "ok", "timestamp_ms": now_ms(), "conveyor": False})
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
