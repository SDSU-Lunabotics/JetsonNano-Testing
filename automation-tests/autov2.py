# send_data.py
from networktables import NetworkTables
import time

ROBORIO_IP = "10.0.8.2"

print(f"Connecting to RoboRIO at {ROBORIO_IP}...")
NetworkTables.initialize(server=ROBORIO_IP)
sd = NetworkTables.getTable("SmartDashboard")

last_command = None

print("Commands:")
print("  Movement: F1.0, B2.0, L0.5, R0.75")
print("  Speed: speed 0.6   (0.0–1.0)")
print("  Turn speed: turnspeed 0.4   (0.0–1.0)")
print("  Enable automation: auto on/off")
print("  Stop now: stop")
print("  Quit: q\n")

while True:
    cmd = input("Command: ").strip().lower()
    if cmd == "q":
        break

    if not cmd:
        continue

    if cmd == "stop":
        sd.putBoolean("Jetson/AutomationEnabled", False)
        sd.putString("Jetson/Command", "")
        last_command = None
        print("Automation disabled + command cleared.")
        continue

    if cmd.startswith("auto "):
        val = cmd.split(" ", 1)[1].strip()
        enable = val in ("on", "true", "1")
        sd.putBoolean("Jetson/AutomationEnabled", enable)
        print(f"Automation {'enabled' if enable else 'disabled'}.")
        continue

    if cmd.startswith("speed "):
        try:
            speed = float(cmd.split(" ", 1)[1])
            speed = max(0.0, min(1.0, speed))
            sd.putNumber("Jetson/Speed", speed)
            print(f"Speed set to {speed}")
        except ValueError:
            print("Invalid speed value.")
        continue

    if cmd.startswith("turnspeed "):
        try:
            speed = float(cmd.split(" ", 1)[1])
            speed = max(0.0, min(1.0, speed))
            sd.putNumber("Jetson/TurnSpeed", speed)
            print(f"Turn speed set to {speed}")
        except ValueError:
            print("Invalid turn speed value.")
        continue

    # movement command (F/B/L/R with seconds)
    if cmd == last_command:
        print("Same as last command, not sending.")
        continue

    sd.putString("Jetson/Command", cmd.upper())
    last_command = cmd
    print(f"Sent: {cmd.upper()}")

    time.sleep(0.05)
