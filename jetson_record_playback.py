# jetson_record_playback.py
from networktables import NetworkTables
import time
import csv

ROBORIO_IP = "10.0.8.2"
NetworkTables.initialize(server=ROBORIO_IP)
sd = NetworkTables.getTable("SmartDashboard")

recording = False
playback = False
path = []

print("Commands: record on/off, save, load, play on/off, clear, q")

def record_tick():
    t = time.time()
    forward = sd.getNumber("Jetson/DriveForward", 0.0)
    turn = sd.getNumber("Jetson/DriveTurn", 0.0)
    path.append([t, forward, turn])

def save_path(filename="recorded_path.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "forward", "turn"])
        writer.writerows(path)

def load_path(filename="recorded_path.csv"):
    loaded = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loaded.append([float(row["t"]), float(row["forward"]), float(row["turn"])])
    return loaded

def to_command(forward, turn, duration):
    if abs(forward) >= abs(turn):
        direction = "F" if forward >= 0 else "B"
    else:
        direction = "R" if turn >= 0 else "L"
    return f"{direction}{duration:.2f}"

last_play_index = 0

while True:
    cmd = input("Command: ").strip().lower()
    if cmd == "q":
        break
    if cmd == "record on":
        recording = True
        print("Recording...")
        continue
    if cmd == "record off":
        recording = False
        print("Stopped recording.")
        continue
    if cmd == "save":
        save_path()
        print("Saved recorded_path.csv")
        continue
    if cmd == "load":
        path = load_path()
        print(f"Loaded {len(path)} points")
        continue
    if cmd == "clear":
        path = []
        print("Cleared path")
        continue
    if cmd == "play on":
        playback = True
        sd.putBoolean("Jetson/AutomationEnabled", True)
        print("Playback enabled")
        continue
    if cmd == "play off":
        playback = False
        sd.putBoolean("Jetson/AutomationEnabled", False)
        sd.putString("Jetson/Command", "")
        print("Playback disabled")
        continue

    # background loop tick
    if recording:
        record_tick()
        print(f"Recorded {len(path)} points")

    if playback and len(path) > 1:
        # play back step-by-step using time deltas
        if last_play_index >= len(path) - 1:
            last_play_index = 0  # loop
        t0, f0, r0 = path[last_play_index]
        t1, _, _ = path[last_play_index + 1]
        duration = max(0.02, min(1.0, t1 - t0))
        cmd_str = to_command(f0, r0, duration)
        sd.putString("Jetson/Command", cmd_str)
        last_play_index += 1
        time.sleep(duration)
