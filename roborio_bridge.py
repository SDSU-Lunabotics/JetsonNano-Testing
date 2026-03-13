import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from networktables import NetworkTables


HOST = "0.0.0.0"
PORT = 8000

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
    "Jetson/AutomationEnabled": "boolean",
    "Jetson/CommandReady": "boolean",
    "Jetson/CommandDuration": "number",
    "Jetson/CommandForward": "number",
    "Jetson/CommandTurn": "number",
    "Jetson/Command": "string",
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

            led_state = bool(data.get("state", False))
            override = bool(data.get("override", True))

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