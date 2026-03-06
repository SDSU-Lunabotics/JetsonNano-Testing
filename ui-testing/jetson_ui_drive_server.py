import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from networktables import NetworkTables

HOST = "0.0.0.0"
PORT = 8000
ROBORIO_IP = "10.0.8.2"

DEFAULT_SPEED = 0.6
DEFAULT_DURATION = 3.0


def clamp(val, lo=-1.0, hi=1.0):
    return max(lo, min(hi, val))


print(f"Connecting to RoboRIO at {ROBORIO_IP}...")
NetworkTables.initialize(server=ROBORIO_IP)
sd = NetworkTables.getTable("SmartDashboard")


def drive_forward(duration=DEFAULT_DURATION, speed=DEFAULT_SPEED):
    duration = max(0.02, float(duration))
    speed = clamp(float(speed))

    sd.putBoolean("Jetson/AutomationEnabled", True)
    sd.putNumber("Jetson/CommandForward", speed)
    sd.putNumber("Jetson/CommandTurn", 0.0)
    sd.putNumber("Jetson/CommandDuration", duration)
    sd.putBoolean("Jetson/CommandReady", True)

    # Give the RoboRIO time to latch the command, then clear the ready flag.
    time.sleep(0.1)
    sd.putBoolean("Jetson/CommandReady", False)

    # Wait for the move to complete, then stop and release automation.
    time.sleep(duration)
    sd.putNumber("Jetson/CommandForward", 0.0)
    sd.putNumber("Jetson/CommandTurn", 0.0)
    sd.putNumber("Jetson/CommandDuration", 0.0)
    sd.putBoolean("Jetson/AutomationEnabled", False)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {"raw": body}

        action = data.get("action", "").lower()
        duration = data.get("duration", DEFAULT_DURATION)
        speed = data.get("speed", DEFAULT_SPEED)

        if self.path == "/drive/forward" or action == "drive_forward":
            threading.Thread(
                target=drive_forward,
                args=(duration, speed),
                daemon=True,
            ).start()
            resp = {
                "status": "ok",
                "action": "drive_forward",
                "duration": duration,
                "speed": speed,
            }
            status = 200
        else:
            resp = {
                "status": "error",
                "message": "Unknown action. Use POST /drive/forward or action=drive_forward.",
            }
            status = 400

        resp_bytes = json.dumps(resp).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Listening on http://{HOST}:{PORT}")
    server.serve_forever()
