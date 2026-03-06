# send_data.py
from networktables import NetworkTables
import time

ROBORIO_IP = "10.0.8.2"

print(f"Connecting to RoboRIO at {ROBORIO_IP}...")
NetworkTables.initialize(server=ROBORIO_IP)
sd = NetworkTables.getTable("SmartDashboard")

last_command = None

print("Enter commands like F1.0, B2.0, L0.5, R0.75")
print("Type 'q' to quit.\n")

while True:
    cmd = input("Command: ").strip()
    if cmd.lower() == "q":
        break

    if not cmd:
        continue

    if cmd == last_command:
        print("Same as last command, not sending.")
        continue

    sd.putString("Jetson/Command", cmd)
    last_command = cmd
    print(f"Sent: {cmd}")

    # tiny sleep so NT flushes
    time.sleep(0.05)
