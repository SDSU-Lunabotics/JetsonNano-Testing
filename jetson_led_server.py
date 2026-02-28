import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from networktables import NetworkTables

HOST = "0.0.0.0"
PORT = 8000
ROBORIO_IP = "10.0.9.2"

DEFAULT_LED_STATE = False


def parse_led_state(value, default=DEFAULT_LED_STATE):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"on", "true", "1", "yes"}:
            return True
        if normalized in {"off", "false", "0", "no"}:
            return False
    return default


print(f"Connecting to RoboRIO at {ROBORIO_IP}...")
NetworkTables.initialize(server=ROBORIO_IP)
sd = NetworkTables.getTable("SmartDashboard")


def wait_for_connection(timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if NetworkTables.isConnected():
            return True
        time.sleep(0.1)
    return NetworkTables.isConnected()


def set_led(state, override=True):
    sd.putBoolean("Jetson/LedOverride", bool(override))
    sd.putBoolean("Jetson/LedState", bool(state))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ping":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"status": "error", "message": "Not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        if self.path != "/led":
            self._send_json(404, {"status": "error", "message": "Use POST /led"})
            return

        led_state = parse_led_state(data.get("state"))
        override = parse_led_state(data.get("override", True), default=True)

        if not wait_for_connection():
            self._send_json(
                503,
                {
                    "status": "error",
                    "message": "RoboRIO not connected",
                },
            )
            return

        threading.Thread(target=set_led, args=(led_state, override), daemon=True).start()

        self._send_json(
            200,
            {
                "status": "ok",
                "led": bool(led_state),
                "override": bool(override),
            },
        )

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
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Listening on http://{HOST}:{PORT}")
    server.serve_forever()
